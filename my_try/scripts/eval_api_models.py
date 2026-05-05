"""
通过 CodeBuddy API（copilot.tencent.com）评测大模型在 CoLA 上的 zero-shot MCC。

目的：扩展 Step 4 的对比矩阵，加入本地跑不动的大模型（闭源 API），
作为我们本地小模型 baseline 的"能力上限"参考。

被测模型（按 my_try.md §Step 4 的扩展需求）：
- hy3-preview              腾讯混元 3.0
- deepseek-v3-0324         DeepSeek V3 (0324 快照)
- claude-opus-4.7          Claude Opus 4.7（当前平台最新旗舰）
- gemini-3.1-flash-lite    Google Gemini 3.1 低延迟版

对齐 my_try/eval_baseline.py 的约定：
- 验证集：cola_data/in_domain_dev.tsv （527 条）
- Prompt：与 PLAIN_TEMPLATE 完全一致（zero-shot，非 thinking）
- 解析：与 parse_answer 完全一致（先 unacceptable 再 acceptable，去 <think>）
- 指标：matthews_corrcoef + 混淆矩阵

API 说明：
- 端点 /v2/chat/completions 仅支持 stream=true
- 某些模型（如 claude-opus-4.6）要求 messages >= 2，本脚本测的 4 个都不需要
- 并发：单任务顺序请求即可（CoLA 只有 527 条，加上限速），避免触发风控

输出：my_try/res_api_{model}.json，字段与本地 res_*.json 对齐，
便于后续合并到 step4_summary.md。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from tqdm import tqdm

# ---- 配置 ----
API_KEY_ENV = "CODEBUDDY_API_KEY"
API_KEY_FALLBACK = "ck_fhl595pwk64g.qltQ6Gk3UcvvJvoUDQJitm0VXfLcY-62u9rFc0ZEmsU"
ENDPOINT = "https://copilot.tencent.com/v2/chat/completions"

DEFAULT_MODELS = [
    "hy3-preview",
    "deepseek-v3-0324",
    "claude-opus-4.7",
    "gemini-3.1-flash-lite",
]

# 每模型的 max_new_tokens 覆盖（--max_new_tokens 作为全局默认）
# - 大部分模型会直接输出 1 个词，16 token 足够
# - hy3-preview 是 thinking 风格，会无视"只输出一个词"的指令，
#   需要 1024 token 才能把推理链讲完并在末尾给出答案
PER_MODEL_MAX_TOKENS = {
    "hy3-preview": 1024,
}

PLAIN_TEMPLATE = (
    'Decide whether the following sentence is grammatically acceptable or not. '
    'If it is grammatically correct, answer "acceptable". '
    'If not, answer "unacceptable". '
    'Only output "acceptable" or "unacceptable", and do not output any other information.\n\n'
    "Sentence: {sentence}\n\n"
    "Your answer:"
)

DATA_PATH = Path("cola_data/in_domain_dev.tsv")
OUT_DIR = Path("my_try/results/step4_api")


# ---- 解析函数（和 eval_baseline.py 的 parse_answer 一致）----
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
    # 兜底：整段里找
    if "unacceptable" in t:
        return "unacceptable"
    if "acceptable" in t:
        return "acceptable"
    return None


# ---- API 调用 ----
def chat_once(
    model: str,
    prompt: str,
    api_key: str,
    max_tokens: int = 16,
    timeout: int = 30,
    temperature: float = 0.0,
    max_retries: int = 3,
    retry_sleep: float = 2.0,
) -> tuple[str, str | None]:
    """发送一条 user 消息，用流式收取完整回复。
    返回 (raw_text, err)；出错时 raw_text 为 ''，err 给失败原因。
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                ENDPOINT,
                headers=headers,
                json=payload,
                timeout=timeout,
                stream=True,
            )
            if resp.status_code != 200:
                body = resp.text[:300]
                last_err = f"HTTP {resp.status_code}: {body}"
                # 4xx 通常不是网络问题，直接返回
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    return "", last_err
                time.sleep(retry_sleep * (attempt + 1))
                continue

            resp.encoding = "utf-8"
            content = ""
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = j.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                c = delta.get("content") or ""
                if c:
                    content += c
                # 某些推理型模型输出在 reasoning_content；本脚本不开 thinking 模式，
                # 但如果 API 返回 reasoning 也拼进去（parse_answer 会剥 </think>）。
                rc = delta.get("reasoning_content") or delta.get("thinking") or ""
                if rc:
                    content += rc
            if not content:
                last_err = "empty stream response"
                time.sleep(retry_sleep * (attempt + 1))
                continue
            return content, None
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(retry_sleep * (attempt + 1))
    return "", last_err


# ---- 评测主流程 ----
def evaluate_model(
    model: str, df: pd.DataFrame, api_key: str, max_tokens: int, sleep_between: float
) -> dict[str, Any]:
    y_true: list[str] = []
    y_pred: list[str] = []
    parse_fail = 0
    api_fail = 0
    first_samples: list[dict[str, Any]] = []

    start = time.time()
    iterator = tqdm(
        list(df.itertuples(index=False)),
        total=len(df),
        desc=model,
        dynamic_ncols=True,
    )
    for idx, row in enumerate(iterator):
        gt = "acceptable" if int(row.label) == 1 else "unacceptable"
        prompt = PLAIN_TEMPLATE.format(sentence=row.text)
        raw, err = chat_once(model, prompt, api_key, max_tokens=max_tokens)
        pred = parse_answer(raw) if raw else None

        if err:
            api_fail += 1
        if pred is None:
            parse_fail += 1
            # 按错位方向归入对立类，与本地脚本一致
            pred_final = "unacceptable" if gt == "acceptable" else "acceptable"
        else:
            pred_final = pred

        y_true.append(gt)
        y_pred.append(pred_final)

        if idx < 10:
            first_samples.append(
                {
                    "sentence": row.text,
                    "true": gt,
                    "raw_output": raw,
                    "parsed": pred,
                    "final_pred": pred_final,
                    "api_error": err,
                }
            )

        # 轻微限速，避免触发风控；默认 0.1s
        if sleep_between > 0:
            time.sleep(sleep_between)

    elapsed = time.time() - start
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(
        y_true, y_pred, labels=["unacceptable", "acceptable"]
    ).tolist()
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    pred_acc_rate = sum(1 for p in y_pred if p == "acceptable") / len(y_pred)

    return {
        "model": model,
        "prompt": "plain",
        "enable_thinking": False,
        "max_new_tokens": max_tokens,
        "source": "codebuddy_api",
        "samples": len(df),
        "mcc": float(mcc),
        "accuracy": float(acc),
        "pred_acc_rate": float(pred_acc_rate),
        "parse_fail": parse_fail,
        "api_fail": api_fail,
        "confusion_matrix": {
            "labels": ["unacceptable", "acceptable"],
            "matrix": cm,
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1],
        },
        "elapsed_sec": round(elapsed, 1),
        "throughput_samples_per_sec": round(len(df) / elapsed, 3) if elapsed > 0 else 0,
        "first_samples": first_samples,
    }


# ---- 命令行入口 ----
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--data", default=str(DATA_PATH))
    p.add_argument("--outdir", default=str(OUT_DIR))
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--sleep", type=float, default=0.08, help="请求之间的秒级间隔")
    p.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 条用于调试")
    p.add_argument("--dry_run", action="store_true", help="只检查 API 连通性后退出")
    args = p.parse_args()

    api_key = os.environ.get(API_KEY_ENV, API_KEY_FALLBACK)
    if not api_key:
        print(f"❌ 未找到 API key（环境变量 {API_KEY_ENV} 未设置）", file=sys.stderr)
        return 2

    # 加载数据
    df = pd.read_csv(
        args.data, sep="\t", header=None,
        names=["source", "label", "first_label", "text"],
    )
    if args.limit > 0:
        df = df.head(args.limit)
    print(f"📊 验证集大小: {len(df)}  (acc={sum(df.label==1)}, unacc={sum(df.label==0)})")

    # 先做一次 ping，确认 key 有效
    probe_model = args.models[0]
    print(f"\n🔎 连通性探测（{probe_model}）...")
    raw, err = chat_once(probe_model, "ping", api_key, max_tokens=4, timeout=20)
    if err:
        print(f"❌ 探测失败: {err}")
        if args.dry_run:
            return 1
    else:
        print(f"✅ 探测 OK, raw={raw!r}")
    if args.dry_run:
        return 0

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # 逐个模型跑
    summary = []
    for model in args.models:
        print(f"\n{'=' * 60}\n🤖 评测模型: {model}\n{'=' * 60}")
        model_max_tokens = PER_MODEL_MAX_TOKENS.get(model, args.max_new_tokens)
        if model_max_tokens != args.max_new_tokens:
            print(f"  ↳ per-model override: max_new_tokens={model_max_tokens}")
        try:
            result = evaluate_model(
                model, df, api_key, model_max_tokens, args.sleep
            )
        except KeyboardInterrupt:
            print("⏹ 用户中断")
            return 130
        # 保存（与本地 res_*.json 字段结构对齐，外层包一层以便混合分析）
        safe_name = model.replace("/", "_").replace(".", "_")
        out_path = Path(args.outdir) / f"res_api_{safe_name}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "prompt": "plain",
                    "enable_thinking": False,
                    "max_new_tokens": model_max_tokens,
                    "source": "codebuddy_api",
                    "results": [result],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        summary.append(
            {
                "model": model,
                "mcc": result["mcc"],
                "accuracy": result["accuracy"],
                "parse_fail": result["parse_fail"],
                "api_fail": result["api_fail"],
                "elapsed": result["elapsed_sec"],
            }
        )
        print(
            f"\n✅ {model}  MCC={result['mcc']:.4f}  Acc={result['accuracy']:.3f}  "
            f"parse_fail={result['parse_fail']}  api_fail={result['api_fail']}  "
            f"t={result['elapsed_sec']}s  → {out_path}"
        )

    # 总表
    print("\n" + "=" * 60 + "\n📋 汇总\n" + "=" * 60)
    print(
        f'{"Model":<28}{"MCC":>8}{"Acc":>8}{"Fail":>6}{"ApiFail":>10}{"Time":>10}'
    )
    print("-" * 70)
    for s in summary:
        print(
            f'{s["model"]:<28}{s["mcc"]:>8.4f}{s["accuracy"]:>8.3f}'
            f'{s["parse_fail"]:>6}{s["api_fail"]:>10}{s["elapsed"]:>9.1f}s'
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
