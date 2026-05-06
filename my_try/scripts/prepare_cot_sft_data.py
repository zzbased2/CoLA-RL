"""
把 DeepSeek 蒸馏的 CoT 数据 → SFT messages JSONL（带 think 思维链）。

输入 :
  my_try/data/cola_train_cot_filtered.jsonl   distill_cot_data.py 产出的过滤后样本
  cola_data/in_domain_dev.tsv                  原始验证集（dev 不蒸馏，用 ground_truth label）

输出 :
  my_try/data/sft_train_cot.jsonl              CoT 训练数据（用 deepseek 的 reasoning）
  my_try/data/sft_val_cot.jsonl                CoT 验证数据（短答案，因为 dev 没蒸馏）

格式 (用 Qwen3 think 模式):
{
  "messages": [
    {"role": "user", "content": "Decide whether...\n\nSentence: ...\n\nYour answer:"},
    {"role": "assistant", "content": "<think>\n{reasoning}\n</think>\n\n{answer}"}
  ]
}

关键设计：
- user 段保持和 PLAIN_TEMPLATE 完全一致（不引入新 prompt）
- assistant 段用 Qwen3 标准的 <think>...</think>{answer} 格式
- SFT 训练时配合 enable_thinking=True（chat_template 不会注入空 think 占位符，
  让模型学会自己生成完整 thinking 链）
- 推理时同样用 enable_thinking=True，从 <think> 开始生成

参考原项目 extract_cot_data.py（用 R1 蒸馏的同样套路）。

用法：
  .venv-cola/bin/python my_try/scripts/prepare_cot_sft_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# 与 PLAIN_TEMPLATE 完全一致的 user prompt（保持训练-评测对齐）
PLAIN_TEMPLATE = (
    'Decide whether the following sentence is grammatically acceptable or not. '
    'If it is grammatically correct, answer "acceptable". '
    'If not, answer "unacceptable". '
    'Only output "acceptable" or "unacceptable", and do not output any other information.\n\n'
    "Sentence: {sentence}\n\n"
    "Your answer:"
)

# Qwen3 think 格式
THINK_FORMAT = "<think>\n{reasoning}\n</think>\n\n{answer}"

DATA_DIR = Path("cola_data")
INPUT_COT = Path("my_try/data/cola_train_cot_filtered.jsonl")
OUT_DIR = Path("my_try/data")
COLS = ["source", "label", "first_label", "text"]


def build_cot_messages(sentence: str, reasoning: str, answer: str) -> dict:
    """带 think 的 messages 格式"""
    return {
        "messages": [
            {"role": "user", "content": PLAIN_TEMPLATE.format(sentence=sentence)},
            {"role": "assistant", "content": THINK_FORMAT.format(
                reasoning=reasoning.strip(), answer=answer
            )},
        ]
    }


def build_short_messages(sentence: str, label: int) -> dict:
    """短答案 messages（dev 没蒸馏，用这个）"""
    answer = "acceptable" if int(label) == 1 else "unacceptable"
    return {
        "messages": [
            {"role": "user", "content": PLAIN_TEMPLATE.format(sentence=sentence)},
            {"role": "assistant", "content": answer},
        ]
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # === 1. 训练集：用蒸馏的 CoT ===
    if not INPUT_COT.exists():
        print(f"❌ 找不到蒸馏数据: {INPUT_COT}")
        print(f"   请先运行: .venv-cola/bin/python my_try/scripts/distill_cot_data.py")
        return

    train_out = OUT_DIR / "sft_train_cot.jsonl"
    n_train = 0
    skipped_short = 0
    with INPUT_COT.open("r", encoding="utf-8") as fin, train_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            reasoning = obj.get("reasoning", "").strip()
            answer = obj.get("answer", "")
            sentence = obj.get("sentence", "")
            # 过滤太短的 reasoning（< 20 字）
            if not reasoning or len(reasoning) < 20:
                skipped_short += 1
                continue
            rec = build_cot_messages(sentence, reasoning, answer)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_train += 1

    print(f"✅ sft_train_cot.jsonl : {n_train} 条（含 think 思维链）")
    if skipped_short > 0:
        print(f"   ⚠️ 跳过 {skipped_short} 条 reasoning 太短的样本")

    # === 2. 验证集：dev 没蒸馏，仍用短答案 ===
    df_val = pd.read_csv(
        DATA_DIR / "in_domain_dev.tsv", sep="\t", header=None, names=COLS,
        dtype={"source": str, "label": int, "first_label": str, "text": str},
        keep_default_na=False,
    )
    val_out = OUT_DIR / "sft_val_cot.jsonl"
    with val_out.open("w", encoding="utf-8") as fout:
        for row in df_val.itertuples(index=False):
            rec = build_short_messages(row.text, row.label)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"✅ sft_val_cot.jsonl   : {len(df_val)} 条（短答案，仅用于 SFT 阶段的 eval_loss）")

    # === 3. 打印 1 条样例 ===
    print(f"\n=== 训练集首条样例（CoT 格式）===")
    with train_out.open("r", encoding="utf-8") as f:
        line = f.readline().strip()
    obj = json.loads(line)
    for m in obj["messages"]:
        print(f"\n[{m['role']}]")
        content = m["content"]
        if len(content) > 600:
            content = content[:600] + "\n... (trunc)"
        print(content)

    print(f"\n输出目录: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
