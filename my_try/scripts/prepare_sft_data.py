"""
将 CoLA 训练集转换为 SFT 所需的 messages JSONL 格式。

输入 : cola_data/in_domain_train.tsv     (8551 条)
       cola_data/in_domain_dev.tsv       (527 条，当 val，也用于后续评测)
输出 : my_try/data/sft_train.jsonl
       my_try/data/sft_val.jsonl

格式 (trl.SFTTrainer 原生支持的 messages 结构):
{
  "messages": [
    {"role": "user", "content": "<原 plain prompt 的 render 结果>"},
    {"role": "assistant", "content": "acceptable" | "unacceptable"}
  ]
}

Prompt 模板与 my_try/eval_baseline.py 的 PLAIN_TEMPLATE 完全一致，
确保训练和评测 prompt 对齐。

按 my_try.md §5.2 方案 A：纯短答案，不带 CoT，训练快且容易评测。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# 与 eval_baseline.py 完全一致的 prompt 模板（避免训练-评测漂移）
PLAIN_TEMPLATE = (
    'Decide whether the following sentence is grammatically acceptable or not. '
    'If it is grammatically correct, answer "acceptable". '
    'If not, answer "unacceptable". '
    'Only output "acceptable" or "unacceptable", and do not output any other information.\n\n'
    "Sentence: {sentence}\n\n"
    "Your answer:"
)

DATA_DIR = Path("cola_data")
OUT_DIR = Path("my_try/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLS = ["source", "label", "first_label", "text"]


def build_messages(text: str, label: int) -> dict:
    answer = "acceptable" if int(label) == 1 else "unacceptable"
    return {
        "messages": [
            {"role": "user", "content": PLAIN_TEMPLATE.format(sentence=text)},
            {"role": "assistant", "content": answer},
        ]
    }


def convert(tsv_name: str, out_name: str) -> int:
    df = pd.read_csv(
        DATA_DIR / tsv_name,
        sep="\t",
        header=None,
        names=COLS,
        dtype={"source": str, "label": int, "first_label": str, "text": str},
        keep_default_na=False,
    )
    out_path = OUT_DIR / out_name
    with out_path.open("w", encoding="utf-8") as f:
        for row in df.itertuples(index=False):
            rec = build_messages(row.text, row.label)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(df)


def main() -> None:
    n_train = convert("in_domain_train.tsv", "sft_train.jsonl")
    n_val = convert("in_domain_dev.tsv", "sft_val.jsonl")
    print(f"✅ sft_train.jsonl : {n_train} 条")
    print(f"✅ sft_val.jsonl   : {n_val} 条")
    print(f"输出目录: {OUT_DIR.resolve()}")

    # 打印前 2 条样例便于人工 review
    print("\n=== 训练集前 2 条样例 ===")
    for i, line in enumerate(
        (OUT_DIR / "sft_train.jsonl").open("r", encoding="utf-8")
    ):
        if i >= 2:
            break
        obj = json.loads(line)
        print(f"\n[{i+1}]")
        for m in obj["messages"]:
            content = m["content"]
            if len(content) > 200:
                content = content[:200] + "... (trunc)"
            print(f"  {m['role']:>9s}: {content!r}")


if __name__ == "__main__":
    main()
