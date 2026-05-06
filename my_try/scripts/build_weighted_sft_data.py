"""
基于 hard mining 结果，构造加权 SFT 训练集 v2：
- 普通样本：保留原本 messages 1 次
- hard 样本（错题 OR prob_true < threshold）：重复 N 次
"""
import argparse
import json
from pathlib import Path

import pandas as pd

PLAIN_TEMPLATE = (
    'Decide whether the following sentence is grammatically acceptable or not. '
    'If it is grammatically correct, answer "acceptable". '
    'If not, answer "unacceptable". '
    'Only output "acceptable" or "unacceptable", and do not output any other information.\n\n'
    "Sentence: {sentence}\n\n"
    "Your answer:"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hard_jsonl", required=True,
                   help="mine_hard_examples.py 的输出")
    p.add_argument("--out", required=True)
    p.add_argument("--prob_threshold", type=float, default=0.6,
                   help="prob_true_label 低于此值算 hard")
    p.add_argument("--repeat", type=int, default=3,
                   help="hard 样本重复次数")
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.hard_jsonl)]
    n_total = len(rows)
    n_hard = sum(1 for r in rows if r["wrong"] or r["prob_true_label"] < args.prob_threshold)
    print(f"输入: {n_total} 条，hard: {n_hard} 条 ({n_hard/n_total*100:.1f}%)")
    print(f"重复策略: 普通 ×1, hard ×{args.repeat}")

    out_records = []
    for r in rows:
        text = r["text"]
        label = r["true_label"]
        answer = "acceptable" if label == 1 else "unacceptable"
        msg = {
            "messages": [
                {"role": "user", "content": PLAIN_TEMPLATE.format(sentence=text)},
                {"role": "assistant", "content": answer},
            ]
        }
        is_hard = r["wrong"] or r["prob_true_label"] < args.prob_threshold
        n_copies = args.repeat if is_hard else 1
        for _ in range(n_copies):
            out_records.append(msg)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"输出: {len(out_records)} 条 → {args.out}")
    print(f"  普通: {n_total - n_hard} × 1 = {n_total - n_hard}")
    print(f"  hard: {n_hard} × {args.repeat} = {n_hard * args.repeat}")


if __name__ == "__main__":
    main()
