"""
Ensemble eval：对多个 LoRA adapter 同时跑 527 条 dev，取 (acc, unacc) 的 logit 平均
然后 argmax 得 ensemble 预测，输出 MCC。

只支持 LoRA adapter（QLoRA 也算 LoRA，base 加载方式可指定）。
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
import torch.nn.functional as F
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


def load_model(base_path: str, adapter_path: str, load_in_4bit: bool):
    print(f"  load base = {base_path} (4bit={load_in_4bit})")
    kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(base_path, **kwargs)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def run_one_model(name, base, adapter, load_in_4bit, prompts, df, batch_size=8):
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    # 尝试两种 tokenization 取较合理 id
    id_acc = tok(" acceptable", add_special_tokens=False).input_ids
    id_unacc = tok(" unacceptable", add_special_tokens=False).input_ids
    print(f"  ' acceptable' tokenize → {id_acc}, ' unacceptable' → {id_unacc}")

    model = load_model(base, adapter, load_in_4bit)

    # 同样的 prompt 文本，但每个模型用各自 tokenizer 重 encode
    enc_prompts = []
    for txt in df["text"]:
        msgs = [{"role": "user", "content": PLAIN_TEMPLATE.format(sentence=txt)}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc_prompts.append(s)

    all_acc_logits = []
    all_unacc_logits = []
    with torch.no_grad():
        for i in tqdm(range(0, len(enc_prompts), batch_size), desc=name):
            batch = enc_prompts[i:i+batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to(model.device)
            out = model(**enc)
            last_idx = enc.attention_mask.sum(dim=1) - 1
            logits = out.logits[torch.arange(out.logits.size(0)), last_idx]  # (B,V)
            # 用第一个 token id（前导空格的 token）
            acc_l = logits[:, id_acc[0]].float().cpu()
            unacc_l = logits[:, id_unacc[0]].float().cpu()
            all_acc_logits.extend(acc_l.tolist())
            all_unacc_logits.extend(unacc_l.tolist())

    # 释放
    del model
    torch.cuda.empty_cache()
    return np.array(all_acc_logits), np.array(all_unacc_logits)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="cola_data/in_domain_dev.tsv")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out", default="my_try/results/step5_lora/res_ensemble_1.7B_E4C_4B_E6.json")
    args = p.parse_args()

    # 读 dev 集
    df = pd.read_csv(
        args.data, sep="\t", header=None,
        names=["source", "label", "first_label", "text"],
        keep_default_na=False,
    )
    print(f"📦 dev: {len(df)} 条")
    true_labels = df["label"].values  # 0 unacc / 1 acc

    # 模型 1：E4-C-r32 (1.7B LoRA)
    name1 = "Qwen3-1.7B-lora-E4-C-r32-r32"
    acc1, unacc1 = run_one_model(
        name1,
        base="model/Qwen3-1.7B",
        adapter="my_try/ckpt/" + name1,
        load_in_4bit=False,
        prompts=None, df=df, batch_size=args.batch_size,
    )

    # 模型 2：E6 (4B QLoRA)
    name2 = "Qwen3-4B-lora-E6-r32"
    acc2, unacc2 = run_one_model(
        name2,
        base="model/Qwen3-4B",
        adapter="my_try/ckpt/" + name2,
        load_in_4bit=True,
        prompts=None, df=df, batch_size=args.batch_size,
    )

    # 单模型预测（重新计算一遍验证 MCC 和原 eval 一致）
    def mcc_from_logits(acc, unacc, name):
        pred = (acc > unacc).astype(int)  # 1=acc 0=unacc
        m = matthews_corrcoef(true_labels, pred)
        cm = confusion_matrix(true_labels, pred, labels=[0, 1])
        print(f"  {name}: MCC={m:.4f}  TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
        return m, pred

    print("\n=== single-model verification ===")
    m1, p1 = mcc_from_logits(acc1, unacc1, name1)
    m2, p2 = mcc_from_logits(acc2, unacc2, name2)

    # ensemble: 各种策略
    print("\n=== ensemble ===")
    results = {"single": {name1: m1, name2: m2}, "ensemble": {}}

    # (a) 直接 logit 相加（等权）
    pred_sum = (acc1 + acc2 > unacc1 + unacc2).astype(int)
    m = matthews_corrcoef(true_labels, pred_sum)
    cm = confusion_matrix(true_labels, pred_sum, labels=[0, 1])
    print(f"  logit_sum:    MCC={m:.4f}  TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    results["ensemble"]["logit_sum"] = m

    # (b) softmax 后概率相加
    paired1 = np.stack([unacc1, acc1], axis=1)
    paired2 = np.stack([unacc2, acc2], axis=1)
    prob1 = torch.softmax(torch.tensor(paired1), dim=1).numpy()
    prob2 = torch.softmax(torch.tensor(paired2), dim=1).numpy()
    prob_avg = (prob1 + prob2) / 2
    pred_p = prob_avg.argmax(axis=1)
    m = matthews_corrcoef(true_labels, pred_p)
    cm = confusion_matrix(true_labels, pred_p, labels=[0, 1])
    print(f"  prob_avg:     MCC={m:.4f}  TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    results["ensemble"]["prob_avg"] = m

    # (c) 投票（必须两个都说 unacc 才说 unacc，否则 acc）
    pred_or = (p1 | p2)  # 任一说 acc 就 acc
    m = matthews_corrcoef(true_labels, pred_or)
    cm = confusion_matrix(true_labels, pred_or, labels=[0, 1])
    print(f"  vote_or_acc:  MCC={m:.4f}  TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    results["ensemble"]["vote_or_acc"] = m

    # (d) 投票（必须两个都说 acc 才说 acc）
    pred_and = (p1 & p2)
    m = matthews_corrcoef(true_labels, pred_and)
    cm = confusion_matrix(true_labels, pred_and, labels=[0, 1])
    print(f"  vote_and_acc: MCC={m:.4f}  TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    results["ensemble"]["vote_and_acc"] = m

    # 保存全量 logits 给后续分析
    out = {
        "models": [name1, name2],
        "n_samples": len(df),
        "results": results,
        "logits": {
            name1: {"acc": acc1.tolist(), "unacc": unacc1.tolist()},
            name2: {"acc": acc2.tolist(), "unacc": unacc2.tolist()},
        },
        "true_labels": true_labels.tolist(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 已保存到 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
