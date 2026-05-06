"""
Hard Example Mining: 用已训好的 LoRA 模型对训练集全量 infer，
挑出模型预测错误或低置信的样本，为后续加权重训做准备。

输出:
  my_try/data/hard_examples_<adapter_name>.jsonl
    每条包含: sentence, true_label, pred_label, wrong, prob_true_label
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True, help="LoRA adapter 目录")
    p.add_argument("--base", default=None)
    p.add_argument("--data", default="cola_data/in_domain_train.tsv")
    p.add_argument("--out", default=None)
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    # 读 base
    adapter_path = Path(args.adapter)
    if args.base is None:
        cfg = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
        base = cfg["base_model_name_or_path"]
    else:
        base = args.base
    out = args.out or f"my_try/data/hard_examples_{adapter_path.name}.jsonl"

    print(f"🧠 基座: {base}")
    print(f"🔧 adapter: {adapter_path}")
    print(f"📦 训练集: {args.data}")
    print(f"💾 输出: {out}")

    # 加载
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    print(f"    加载耗时 {time.time() - t0:.1f}s，显存 {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # 读训练集
    df = pd.read_csv(
        args.data, sep="\t", header=None,
        names=["source", "label", "first_label", "text"],
        dtype={"source": str, "label": int, "first_label": str, "text": str},
        keep_default_na=False,
    )
    print(f"    训练集: {len(df)} 条")

    # 预计算 acceptable / unacceptable 的首 token id
    id_acc = tok(" acceptable", add_special_tokens=False).input_ids[0]
    id_unacc = tok(" unacceptable", add_special_tokens=False).input_ids[0]
    # Qwen3 tokenizer 里 acceptable 和 unacceptable 开头可能不同，也准备不带空格版本
    id_acc2 = tok("acceptable", add_special_tokens=False).input_ids[0]
    id_unacc2 = tok("unacceptable", add_special_tokens=False).input_ids[0]
    print(f"    token ids: ' acceptable'={id_acc} ' unacceptable'={id_unacc} "
          f"'acceptable'={id_acc2} 'unacceptable'={id_unacc2}")

    # 构造 prompt（用 chat_template）
    prompts = []
    for _, row in df.iterrows():
        msgs = [{"role": "user", "content": PLAIN_TEMPLATE.format(sentence=row["text"])}]
        try:
            p_ = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,  # Qwen3 关闭 thinking，直接预测 acceptable/unacceptable
            )
        except TypeError:
            p_ = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(p_)

    # batch 前向，拿最后一个 token 的 logit，对 acc / unacc 取 softmax
    results = []
    bsz = args.batch_size
    n_wrong = 0
    n_total = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), bsz), desc="infer"):
            batch_prompts = prompts[i:i+bsz]
            batch_labels = df["label"].iloc[i:i+bsz].tolist()
            batch_texts = df["text"].iloc[i:i+bsz].tolist()

            enc = tok(batch_prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(model.device)
            out_ = model(**enc)
            # left-padding 时，所有样本的最后一个真实 token 都在 seq_len-1 位置
            # （right-padding 才需要用 attention_mask.sum()-1 找）
            last_logits = out_.logits[:, -1, :]  # (B, V)

            # 组合 acc / unacc 的 logit（取两种 tokenization 中的 max）
            acc_logit = torch.maximum(last_logits[:, id_acc], last_logits[:, id_acc2])
            unacc_logit = torch.maximum(last_logits[:, id_unacc], last_logits[:, id_unacc2])
            paired = torch.stack([unacc_logit, acc_logit], dim=1)  # (B, 2)  0=unacc 1=acc
            probs = F.softmax(paired.float(), dim=1)  # (B, 2)
            preds = probs.argmax(dim=1).tolist()
            probs_list = probs.tolist()

            for txt, true_lbl, pred, pr in zip(batch_texts, batch_labels, preds, probs_list):
                wrong = int(pred != true_lbl)
                n_wrong += wrong
                n_total += 1
                results.append({
                    "text": txt,
                    "true_label": int(true_lbl),
                    "pred_label": int(pred),
                    "wrong": bool(wrong),
                    "prob_unacc": pr[0],
                    "prob_acc": pr[1],
                    "prob_true_label": pr[true_lbl],
                    "margin": abs(pr[1] - pr[0]),  # 置信度：越小越"难"
                })

    acc_rate = 1 - n_wrong / n_total
    print(f"\n📊 训练集整体: {n_total} 条，正确 {n_total-n_wrong} 错 {n_wrong}，"
          f"accuracy = {acc_rate:.4f}")

    # 按 prob_true_label 升序排（最难的在前）
    results.sort(key=lambda x: x["prob_true_label"])

    # 写盘
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"💾 已写入 {out}")

    # 打印一些统计
    hard = [r for r in results if r["wrong"]]
    low_conf = [r for r in results if r["prob_true_label"] < 0.6]
    print(f"    错题: {len(hard)}")
    print(f"    低置信 (prob_true<0.6): {len(low_conf)}")
    print(f"    最难的 5 条:")
    for r in results[:5]:
        print(f"      prob_true={r['prob_true_label']:.3f} wrong={r['wrong']} "
              f"true={'acc' if r['true_label']==1 else 'unacc'}  {r['text'][:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
