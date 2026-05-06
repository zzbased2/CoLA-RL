"""
Sampling-based 评测：解决 GRPO mode collapse 在贪心解码下的"软崩"问题。

背景（见 docs/step6_summary.md §5.2）：
  GRPO 训练时用 sampling，模型可能学到 P(unacc)=0.45, P(acc)=0.55 这种概率分布。
  贪心评测时永远选 argmax = acc → MCC=0
  采样评测可以揭示模型真实学到的概率。

做法：
  对每条 dev 样本：
    1. do_sample=True, T=1.0, 生成 N 次（默认 5）
    2. parse 出 N 个答案
    3. majority vote 取众数
    4. 算 MCC

用法（从项目根）：
  .venv-cola/bin/python my_try/scripts/eval_lora_sampling.py \
      --adapter my_try/ckpt/Qwen3-0.6B-grpo-E1-GRPO-v3-r16 \
      --num_samples 5 --temperature 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PLAIN_TEMPLATE = (
    'Decide whether the following sentence is grammatically acceptable or not. '
    'If it is grammatically correct, answer "acceptable". '
    'If not, answer "unacceptable". '
    'Only output "acceptable" or "unacceptable", and do not output any other information.\n\n'
    "Sentence: {sentence}\n\n"
    "Your answer:"
)


def parse_answer(text: str) -> str | None:
    t = text
    if "</think>" in t:
        t = t.split("</think>")[-1]
    t = t.strip().lower()
    non_empty_lines = [ln for ln in t.splitlines() if ln.strip()]
    if non_empty_lines:
        tail = non_empty_lines[-1]
        if "unacceptable" in tail:
            return "unacceptable"
        if "acceptable" in tail:
            return "acceptable"
    if "unacceptable" in t:
        return "unacceptable"
    if "acceptable" in t:
        return "acceptable"
    return None


def majority_vote(preds: list[str | None], gt: str) -> tuple[str, dict[str, int]]:
    """对 N 次采样的预测做投票，返回 (final_pred, vote_count)"""
    valid = [p for p in preds if p is not None]
    if not valid:
        # 全部解析失败
        return ("unacceptable" if gt == "acceptable" else "acceptable",
                {"acceptable": 0, "unacceptable": 0, "fail": len(preds)})
    counter = Counter(valid)
    counter["fail"] = len(preds) - len(valid)
    # most_common 在 acc/unacc 平票时按出现顺序，加一个 deterministic tiebreak
    if counter["acceptable"] == counter["unacceptable"]:
        # 平票时倾向于 unacceptable（比贪心更激进，让它有机会答对少数类）
        return "unacceptable", dict(counter)
    return counter.most_common(1)[0][0], dict(counter)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True)
    p.add_argument("--base", default=None)
    p.add_argument("--data", default="cola_data/in_domain_dev.tsv")
    p.add_argument("--out", default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--num_samples", type=int, default=5,
                   help="每条样本采样几次做投票（默认 5）")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    adapter_path = Path(args.adapter)
    if args.base is None:
        cfg = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
        base = cfg["base_model_name_or_path"]
    else:
        base = args.base
    print(f"🧠 基座模型: {base}")
    print(f"🔧 LoRA adapter: {adapter_path}")
    print(f"🎲 采样配置: N={args.num_samples}, T={args.temperature}, top_p={args.top_p}")

    tok = AutoTokenizer.from_pretrained(adapter_path)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model = model.merge_and_unload()
    model.eval()
    print(f"   加载耗时: {time.time() - t0:.1f}s；显存: "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")

    df = pd.read_csv(
        args.data, sep="\t", header=None,
        names=["source", "label", "first_label", "text"],
        dtype={"source": str, "label": int, "first_label": str, "text": str},
        keep_default_na=False,
    )
    print(f"📊 验证集: {len(df)}  (acc={sum(df.label==1)}, unacc={sum(df.label==0)})")

    def build_batch_inputs(sentences):
        batch_messages = [
            [{"role": "user", "content": PLAIN_TEMPLATE.format(sentence=s)}]
            for s in sentences
        ]
        enc = tok.apply_chat_template(
            batch_messages, add_generation_prompt=True,
            enable_thinking=False, padding=True,
            return_tensors="pt", return_dict=True,
        )
        return enc["input_ids"].to("cuda"), enc["attention_mask"].to("cuda")

    y_true, y_pred = [], []
    parse_fail_total = 0
    first_samples = []

    start = time.time()
    for i in tqdm(range(0, len(df), args.batch_size), desc="eval"):
        chunk = df.iloc[i : i + args.batch_size]
        sentences = list(chunk.text.values)
        input_ids, attn_mask = build_batch_inputs(sentences)

        # 一次 generate 用 num_return_sequences 出 N 个采样
        with torch.no_grad():
            out = model.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.num_samples,
                pad_token_id=tok.eos_token_id,
            )
        # out shape: (batch * num_samples, total_len)
        gen = out[:, input_ids.shape[-1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        # 把 batch * num_samples 的扁平结果重塑为 batch x num_samples
        bsz = input_ids.shape[0]
        nsamp = args.num_samples
        # transformers 返回顺序：sample0_seq0, sample0_seq1, ..., sample1_seq0, ...
        # 对每个 i 的 N 次采样取 texts[i*nsamp : (i+1)*nsamp]
        for j in range(bsz):
            sentence = sentences[j]
            label = int(chunk.iloc[j].label)
            gt = "acceptable" if label == 1 else "unacceptable"
            j_texts = texts[j * nsamp : (j + 1) * nsamp]
            j_preds = [parse_answer(t) for t in j_texts]
            final, votes = majority_vote(j_preds, gt)
            y_true.append(gt)
            y_pred.append(final)
            parse_fail_total += sum(1 for p in j_preds if p is None)

            if len(first_samples) < 10:
                first_samples.append({
                    "sentence": sentence,
                    "true": gt,
                    "raw_outputs": j_texts,
                    "parsed": j_preds,
                    "votes": votes,
                    "final_pred": final,
                })

    elapsed = time.time() - start
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=["unacceptable", "acceptable"]).tolist()
    acc = sum(1 for t, pp in zip(y_true, y_pred) if t == pp) / len(y_true)
    pred_acc_rate = sum(1 for pp in y_pred if pp == "acceptable") / len(y_pred)
    parse_fail_rate = parse_fail_total / (len(df) * args.num_samples)

    result = {
        "adapter": str(adapter_path),
        "base_model": base,
        "mode": "sampling_vote",
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "samples": len(df),
        "mcc": float(mcc),
        "accuracy": float(acc),
        "pred_acc_rate": float(pred_acc_rate),
        "parse_fail_rate": float(parse_fail_rate),
        "confusion_matrix": {
            "labels": ["unacceptable", "acceptable"], "matrix": cm,
            "tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1],
        },
        "elapsed_sec": round(elapsed, 1),
        "first_samples": first_samples,
    }

    if args.out is None:
        args.out = f"my_try/results/step6_grpo/res_lora_sampling_{adapter_path.name}.json"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {"source": "lora_eval_sampling", "results": [result]},
            f, ensure_ascii=False, indent=2,
        )

    print("\n" + "=" * 60)
    print(f"✅ Sampling 评测完成: {adapter_path.name}")
    print(f"   N={args.num_samples}, T={args.temperature}")
    print(f"   MCC          : {mcc:.4f}")
    print(f"   Accuracy     : {acc:.4f}")
    print(f"   pred_acc_rate: {pred_acc_rate:.4f}")
    print(f"   parse_fail   : {parse_fail_rate:.4f} ({parse_fail_total}/{len(df)*args.num_samples})")
    print(f"   耗时         : {elapsed:.1f}s")
    print(f"   混淆矩阵     : {cm}  (unacc, acc)")
    print(f"   结果文件     : {args.out}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
