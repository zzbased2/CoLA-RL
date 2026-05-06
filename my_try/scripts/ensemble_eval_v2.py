"""
Ensemble eval (V2): 用各模型生成式输出 (parse 后的 acceptable/unacceptable) 做 vote。
注意：parse_answer 在原 eval_lora 中处理 chat template 输出，不能简单看 logit。

输出每条样本的 raw_output / parsed，并做 4 种集成。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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


def run_one_model(name, base, adapter, load_in_4bit, df, batch_size=16, max_new_tokens=8):
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    enc_prompts = []
    for txt in df["text"]:
        msgs = [{"role": "user", "content": PLAIN_TEMPLATE.format(sentence=txt)}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc_prompts.append(s)

    preds = []
    raws = []
    with torch.no_grad():
        for i in tqdm(range(0, len(enc_prompts), batch_size), desc=name):
            batch = enc_prompts[i:i+batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to(model.device)
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=1.0, top_p=1.0,
                pad_token_id=tok.pad_token_id,
            )
            new_ids = gen[:, enc.input_ids.shape[1]:]
            for ids in new_ids:
                txt = tok.decode(ids, skip_special_tokens=True)
                raws.append(txt)
                p = parse_answer(txt)
                if p is None:
                    preds.append(-1)
                elif p == "acceptable":
                    preds.append(1)
                else:
                    preds.append(0)

    del model
    torch.cuda.empty_cache()
    return np.array(preds), raws


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="cola_data/in_domain_dev.tsv")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out", default="my_try/results/step5_lora/res_ensemble_v2.json")
    args = p.parse_args()

    df = pd.read_csv(
        args.data, sep="\t", header=None,
        names=["source", "label", "first_label", "text"],
        keep_default_na=False,
    )
    print(f"📦 dev: {len(df)} 条")
    true = df["label"].values

    name1 = "Qwen3-1.7B-lora-E4-C-r32-r32"
    p1, r1 = run_one_model(
        name1, "model/Qwen3-1.7B",
        "my_try/ckpt/" + name1, False, df, args.batch_size
    )

    name2 = "Qwen3-4B-lora-E6-r32"
    p2, r2 = run_one_model(
        name2, "model/Qwen3-4B",
        "my_try/ckpt/" + name2, True, df, args.batch_size
    )

    # 单模型验证
    def mcc(pred, name):
        valid = (pred >= 0)
        if valid.sum() < len(pred):
            print(f"  {name}: parse_fail={len(pred)-valid.sum()}")
        m = matthews_corrcoef(true[valid], pred[valid])
        cm = confusion_matrix(true[valid], pred[valid], labels=[0,1])
        print(f"  {name}: MCC={m:.4f} TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]} "
              f"unacc_recall={cm[0,0]/(cm[0,0]+cm[0,1])*100:.1f}% acc_recall={cm[1,1]/(cm[1,0]+cm[1,1])*100:.1f}%")
        return m

    print("\n=== 单模型验证 ===")
    m1 = mcc(p1, name1)
    m2 = mcc(p2, name2)

    print("\n=== Ensemble 策略 ===")
    # 处理 parse_fail：parse 失败按 acceptable 处理（多数类）
    p1_fb = np.where(p1 == -1, 1, p1)
    p2_fb = np.where(p2 == -1, 1, p2)

    # OR: 任一说 acc 就 acc
    pred_or = (p1_fb | p2_fb)
    m_or = matthews_corrcoef(true, pred_or)
    cm = confusion_matrix(true, pred_or, labels=[0,1])
    print(f"  vote_OR (任一 acc → acc):  MCC={m_or:.4f} TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    # AND: 都说 acc 才 acc
    pred_and = (p1_fb & p2_fb)
    m_and = matthews_corrcoef(true, pred_and)
    cm = confusion_matrix(true, pred_and, labels=[0,1])
    print(f"  vote_AND (双 acc → acc):   MCC={m_and:.4f} TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    # 取较强 unacc 模型主导（如果 1.7B 说 unacc，且 4B 也犹豫，按 1.7B；否则 4B）
    # 这里简化：1.7B 和 4B 不一致时，看哪个模型本身在该类别召回更高
    # 1.7B unacc_recall 80% > 4B 67%  → 不一致时听 1.7B
    # 1.7B acc_recall 87.7% < 4B 94.5% → 不一致时听 4B
    # 综合：判 acc 时听 4B，判 unacc 时听 1.7B
    pred_smart = np.where(p1_fb == 0, 0, p2_fb)  # 1.7B 说 unacc 就 unacc，否则用 4B
    m_smart = matthews_corrcoef(true, pred_smart)
    cm = confusion_matrix(true, pred_smart, labels=[0,1])
    print(f"  smart (1.7B说unacc优先):  MCC={m_smart:.4f} TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    # 反向 smart：4B 说 unacc 就 unacc
    pred_smart2 = np.where(p2_fb == 0, 0, p1_fb)
    m_smart2 = matthews_corrcoef(true, pred_smart2)
    cm = confusion_matrix(true, pred_smart2, labels=[0,1])
    print(f"  smart2 (4B说unacc优先):   MCC={m_smart2:.4f} TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    # save
    out = {
        "models": [name1, name2],
        "single_mcc": {name1: m1, name2: m2},
        "ensemble_mcc": {
            "OR_acc": m_or, "AND_acc": m_and,
            "smart_1.7B_unacc_priority": m_smart,
            "smart_4B_unacc_priority": m_smart2,
        },
        "preds": {name1: p1.tolist(), name2: p2.tolist()},
        "true_labels": true.tolist(),
        "first_raws": [{"text": t, "true": int(tl), "r1": r1_, "r2": r2_}
                       for t, tl, r1_, r2_ in zip(df["text"][:20], true[:20], r1[:20], r2[:20])]
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 {args.out}")

if __name__ == "__main__":
    sys.exit(main())
