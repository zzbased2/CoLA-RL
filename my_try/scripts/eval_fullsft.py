"""
评测全参 SFT 训练后的模型在 CoLA dev 集上的 MCC。

与 eval_lora.py 的区别：这里直接从 output_dir 加载整个模型（不是 base + adapter）。
使用和 eval_baseline.py 完全一致的 prompt + parse 逻辑，保证可比性。

用法:
  .venv-cola/bin/python my_try/eval_fullsft.py \\
      --model my_try/ckpt/Qwen3-0.6B-fullsft-E1-FULL
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="全参 SFT 产出目录")
    p.add_argument("--data", default="cola_data/in_domain_dev.tsv")
    p.add_argument("--out", default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=8)
    args = p.parse_args()

    model_path = Path(args.model)
    print(f"🧠 加载全参 SFT 模型: {model_path}")

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only 生成必备

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda"
    )
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

    def build_batch_inputs(sentences: list[str]):
        batch_messages = [
            [{"role": "user", "content": PLAIN_TEMPLATE.format(sentence=s)}]
            for s in sentences
        ]
        enc = tok.apply_chat_template(
            batch_messages, add_generation_prompt=True, enable_thinking=False,
            padding=True, return_tensors="pt", return_dict=True,
        )
        return enc["input_ids"].to("cuda"), enc.get("attention_mask").to("cuda")

    y_true, y_pred, parse_fail = [], [], 0
    first_samples = []
    total_new_tokens = 0
    start = time.time()
    for i in tqdm(range(0, len(df), args.batch_size), desc="eval"):
        chunk = df.iloc[i : i + args.batch_size]
        sentences = list(chunk.text.values)
        input_ids, attn_mask = build_batch_inputs(sentences)
        with torch.no_grad():
            out = model.generate(
                input_ids, attention_mask=attn_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id,
            )
        gen = out[:, input_ids.shape[-1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for j, (sentence, raw) in enumerate(zip(sentences, texts)):
            lbl = int(chunk.iloc[j].label)
            gt = "acceptable" if lbl == 1 else "unacceptable"
            pred = parse_answer(raw)
            if pred is None:
                parse_fail += 1
                final = "unacceptable" if gt == "acceptable" else "acceptable"
            else:
                final = pred
            y_true.append(gt)
            y_pred.append(final)
            total_new_tokens += (gen[j] != tok.eos_token_id).sum().item()
            if len(first_samples) < 10:
                first_samples.append({
                    "sentence": sentence, "true": gt,
                    "raw_output": raw, "parsed": pred, "final_pred": final,
                })

    elapsed = time.time() - start
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=["unacceptable", "acceptable"]).tolist()
    acc = sum(1 for t, pv in zip(y_true, y_pred) if t == pv) / len(y_true)

    result = {
        "model": str(model_path),
        "mode": "full_sft",
        "prompt": "plain",
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "samples": len(df),
        "mcc": float(mcc),
        "accuracy": float(acc),
        "pred_acc_rate": sum(1 for pv in y_pred if pv == "acceptable") / len(y_pred),
        "parse_fail": parse_fail,
        "avg_new_tokens": total_new_tokens / len(df),
        "confusion_matrix": {
            "labels": ["unacceptable", "acceptable"], "matrix": cm,
            "tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1],
        },
        "elapsed_sec": round(elapsed, 1),
        "first_samples": first_samples,
    }

    if args.out is None:
        args.out = f"my_try/results/step5_fullsft/res_fullsft_{model_path.name}.json"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {"source": "fullsft_eval", "prompt": "plain",
             "max_new_tokens": args.max_new_tokens, "results": [result]},
            f, ensure_ascii=False, indent=2,
        )

    print("\n" + "=" * 60)
    print(f"✅ 全参 SFT 评测完成: {model_path.name}")
    print(f"   MCC       : {mcc:.4f}")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   Parse fail: {parse_fail}/{len(df)}")
    print(f"   耗时      : {elapsed:.1f}s")
    print(f"   混淆矩阵  : {cm}")
    print(f"   结果文件  : {args.out}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
