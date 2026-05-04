"""
Baseline 零样本评测脚本（my_try.md §Step 3）

对比 Qwen3-0.6B / Qwen3-1.7B / Qwen3.5-0.8B / Qwen3.5-2B 四个模型
在 CoLA in_domain_dev（527 条）上的 MCC。

相对 my_try.md §3.3 文档版的修正（详见 environment.md §5 踩坑 1）：
- transformers 5.x 的 tokenizer.apply_chat_template(return_tensors="pt")
  返回 BatchEncoding 而非 Tensor，不能直接 .shape / 喂 model.generate。
  这里统一加上 return_dict=True，然后从 dict 里取 input_ids/attention_mask。
- AutoModelForCausalLM.from_pretrained 的 torch_dtype 在 5.x 已 deprecated，
  改用 dtype=...。

支持 4 种 prompt 模式（--prompt）：
- plain    : 原 zero-shot prompt（Step 3 已有）
- fewshot  : 原 prompt 前置 4 条均衡示例（2 正 2 负），纯 in-context
- thinking : plain prompt + enable_thinking=True + max_new_tokens=1024
             （Qwen3 原生 think 通道，模型会先输出 <think>…</think> 再给答案）
- cot      : plain prompt 中追加 "Let's think step by step." 诱导 CoT，
             不开 thinking token，max_new_tokens=256
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


PLAIN_TEMPLATE = (
    'Decide whether the following sentence is grammatically acceptable or not. '
    'If it is grammatically correct, answer "acceptable". '
    'If not, answer "unacceptable". '
    'Only output "acceptable" or "unacceptable", and do not output any other information.\n\n'
    "Sentence: {sentence}\n\n"
    "Your answer:"
)

# 从 in_domain_train.tsv 人工挑的 4 个短、典型示例（2 正 2 负）
FEWSHOT_EXAMPLES = [
    ("I'll fix you a drink.", "acceptable"),
    ("They drank the pub.", "unacceptable"),
    ("The pond froze solid.", "acceptable"),
    ("We yelled ourselves.", "unacceptable"),
]


def _fewshot_block() -> str:
    lines = ["Here are a few examples:\n"]
    for s, lab in FEWSHOT_EXAMPLES:
        lines.append(f"Sentence: {s}\nYour answer: {lab}\n")
    lines.append("Now decide for the following sentence.\n")
    return "\n".join(lines)


FEWSHOT_TEMPLATE = (
    'Decide whether the following sentence is grammatically acceptable or not. '
    'If it is grammatically correct, answer "acceptable". '
    'If not, answer "unacceptable". '
    'Only output "acceptable" or "unacceptable", and do not output any other information.\n\n'
    + _fewshot_block()
    + "\nSentence: {sentence}\n\nYour answer:"
)

COT_TEMPLATE = (
    'Decide whether the following sentence is grammatically acceptable or not. '
    'Think step by step about the grammar (subject-verb agreement, argument structure, '
    'word order, etc.), then on the LAST line output ONLY one word: '
    '"acceptable" or "unacceptable".\n\n'
    "Sentence: {sentence}\n\n"
    "Let's think step by step."
)

PROMPT_TEMPLATES = {
    "plain": PLAIN_TEMPLATE,
    "fewshot": FEWSHOT_TEMPLATE,
    "thinking": PLAIN_TEMPLATE,  # 与 plain 相同，但打开 enable_thinking
    "cot": COT_TEMPLATE,
}
# 不同 prompt 模式对应的合理默认生成长度
DEFAULT_MAX_NEW_TOKENS = {
    "plain": 32,
    "fewshot": 32,
    "thinking": 1024,
    "cot": 256,
}


def parse_answer(text: str) -> str | None:
    """从模型输出中抽取 acceptable / unacceptable。

    注意顺序：先判 'unacceptable'，再判 'acceptable'，因为后者是前者的子串。
    若输出包含 </think>，只取其后部分（去除思考段只保留最终回答）。
    对 CoT 多行输出，取最后一个非空行做判断，避免中途出现的词干扰。
    """
    t = text
    if "</think>" in t:
        t = t.split("</think>")[-1]
    t = t.strip().lower()
    # 对 CoT 类多行输出，仅取最后一个非空行
    non_empty_lines = [ln for ln in t.splitlines() if ln.strip()]
    if non_empty_lines:
        tail = non_empty_lines[-1]
        if "unacceptable" in tail:
            return "unacceptable"
        if "acceptable" in tail:
            return "acceptable"
    # 回退：整段里找
    if "unacceptable" in t:
        return "unacceptable"
    if "acceptable" in t:
        return "acceptable"
    return None


def build_inputs(tok, sentences: list[str], enable_thinking: bool, prompt_mode: str):
    """构造批量输入（兼容 transformers 4.x / 5.x）。支持 1 条或 N 条。"""
    template = PROMPT_TEMPLATES[prompt_mode]
    batch_messages = [
        [{"role": "user", "content": template.format(sentence=s)}] for s in sentences
    ]
    kwargs = dict(
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    )
    try:
        enc = tok.apply_chat_template(
            batch_messages, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        enc = tok.apply_chat_template(batch_messages, **kwargs)
    return enc


def evaluate(
    model_path: str,
    data_path: str,
    prompt_mode: str,
    enable_thinking: bool,
    max_new_tokens: int,
    batch_size: int = 1,
    limit: int | None = None,
) -> dict:
    print(f"\n{'=' * 60}")
    print(f"Model    : {model_path}")
    print(f"Prompt   : {prompt_mode}")
    print(f"Thinking : {enable_thinking}")
    print(f"MaxNewTok: {max_new_tokens}")
    print(f"BatchSize: {batch_size}")
    print(f"{'=' * 60}")

    tok = AutoTokenizer.from_pretrained(model_path)
    # padding 方向统一为 left，batch generate 必须
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    mdl.eval()

    df = pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=["source", "label", "first_label", "text"],
        dtype={"source": str, "label": int, "first_label": str, "text": str},
        keep_default_na=False,
    )
    if limit is not None:
        df = df.head(limit).reset_index(drop=True)
    print(f"验证集大小: {len(df)}")

    y_true: list[str] = []
    y_pred: list[str] = []
    samples_dump: list[dict] = []
    parse_fail = 0
    gen_tokens_sum = 0
    start = time.time()

    sentences = df["text"].tolist()
    labels = df["label"].tolist()
    pbar = tqdm(
        range(0, len(df), batch_size),
        total=(len(df) + batch_size - 1) // batch_size,
        desc="eval",
        mininterval=2.0,
    )
    for i in pbar:
        chunk_sents = sentences[i : i + batch_size]
        chunk_labels = labels[i : i + batch_size]
        enc = build_inputs(tok, chunk_sents, enable_thinking, prompt_mode)
        input_ids = enc["input_ids"].to("cuda")
        attn_mask = enc["attention_mask"].to("cuda")
        prompt_len = input_ids.shape[-1]

        with torch.no_grad():
            out = mdl.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = out[:, prompt_len:]
        resps = tok.batch_decode(new_tokens, skip_special_tokens=True)
        for j, (sent, lab, resp) in enumerate(zip(chunk_sents, chunk_labels, resps)):
            # 新 token 数（按非 pad 粗略计；细算要排除右侧 padding，此处忽略）
            n_new = int((new_tokens[j] != tok.pad_token_id).sum().item())
            gen_tokens_sum += n_new
            pred = parse_answer(resp)
            true_label = "acceptable" if int(lab) == 1 else "unacceptable"
            if pred is None:
                parse_fail += 1
                fallback = "unacceptable" if true_label == "acceptable" else "acceptable"
                pred_final = fallback
            else:
                pred_final = pred
            y_true.append(true_label)
            y_pred.append(pred_final)
            if len(samples_dump) < 10:
                samples_dump.append(
                    {
                        "sentence": sent,
                        "true": true_label,
                        "raw_output": resp,
                        "new_tokens": n_new,
                        "parsed": pred,
                        "final_pred": pred_final,
                    }
                )

    elapsed = time.time() - start
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=["unacceptable", "acceptable"])
    # confusion_matrix 行 = true, 列 = pred
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    acc = (tp + tn) / max(1, tp + tn + fp + fn)

    result = {
        "model": model_path,
        "prompt": prompt_mode,
        "enable_thinking": enable_thinking,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "samples": len(df),
        "mcc": float(mcc),
        "accuracy": float(acc),
        "parse_fail": parse_fail,
        "avg_new_tokens": round(gen_tokens_sum / max(1, len(df)), 2),
        "confusion_matrix": {
            "labels": ["unacceptable", "acceptable"],
            "matrix": cm.tolist(),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "elapsed_sec": round(elapsed, 1),
        "throughput_samples_per_sec": round(len(df) / max(elapsed, 1e-9), 3),
        "first_samples": samples_dump,
    }

    print(f"\nMCC      : {mcc:.4f}")
    print(f"Accuracy : {acc:.4f}")
    print(f"ParseFail: {parse_fail}/{len(df)}")
    print(f"AvgNewTok: {result['avg_new_tokens']}")
    print(f"Confusion (rows=true, cols=pred, labels=[unacc, acc]):\n{cm}")
    print(f"Elapsed  : {elapsed:.1f}s ({result['throughput_samples_per_sec']} it/s)")

    # 释放显存
    del mdl
    del tok
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="cola_data/in_domain_dev.tsv")
    parser.add_argument("--out", default="my_try/baseline_results.json")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "model/Qwen3-0.6B",
            "model/Qwen3-1.7B",
            "model/Qwen3.5-0.8B",
            "model/Qwen3.5-2B",
        ],
    )
    parser.add_argument(
        "--prompt",
        choices=list(PROMPT_TEMPLATES.keys()),
        default="plain",
        help="prompt 模式：plain / fewshot / thinking / cot",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="默认按 prompt 模式自动选：plain/fewshot=32, cot=256, thinking=1024",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="只跑前 N 条做 smoke test"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批大小。thinking/cot 模式建议 8-16，plain/fewshot 保持 1 即可",
    )
    args = parser.parse_args()

    # thinking 模式强制打开 enable_thinking；其他模式关闭
    enable_thinking = args.prompt == "thinking"
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else DEFAULT_MAX_NEW_TOKENS[args.prompt]
    )

    results: list[dict] = []
    for m in args.models:
        if not Path(m).exists():
            print(f"[skip] {m} 不存在")
            results.append({"model": m, "error": "path not found"})
            continue
        try:
            results.append(
                evaluate(
                    m,
                    args.data,
                    args.prompt,
                    enable_thinking,
                    max_new_tokens,
                    batch_size=args.batch_size,
                    limit=args.limit,
                )
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            results.append({"model": m, "error": f"{type(e).__name__}: {e}"})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "prompt": args.prompt,
                "enable_thinking": enable_thinking,
                "max_new_tokens": max_new_tokens,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # 打印总表
    print(f"\n{'=' * 60}")
    print(f"汇总：prompt={args.prompt}")
    print(f"{'=' * 60}")
    print(f"{'model':<32}{'mcc':<10}{'acc':<10}{'fail':<8}{'avgtok':<10}{'sec':<8}")
    for r in results:
        if "error" in r:
            print(f"{r['model']:<32}ERROR: {r['error']}")
            continue
        print(
            f"{r['model']:<32}"
            f"{r['mcc']:<10.4f}"
            f"{r['accuracy']:<10.4f}"
            f"{r['parse_fail']:<8}"
            f"{r['avg_new_tokens']:<10.1f}"
            f"{r['elapsed_sec']:<8.1f}"
        )
    print(f"\n结果已保存: {args.out}")