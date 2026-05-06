"""
用 DeepSeek-V3-0324（通过 CodeBuddy API）蒸馏 CoLA 训练集，得到带 CoT 的 SFT 数据。

输入: cola_data/in_domain_train.tsv         8551 条
输出: my_try/data/cola_train_cot_raw.jsonl  原始蒸馏结果（含 deepseek 的 reasoning + answer）
     my_try/data/cola_train_cot_filtered.jsonl  过滤后只保留 deepseek 答对 GT 的样本

每条记录格式：
{
  "sentence": "...",
  "label": 1,
  "ground_truth": "acceptable",
  "reasoning": "Let's analyze... subject-verb agreement is correct, ...",
  "answer": "acceptable",       # deepseek 自己给的答案
  "match": true,                # answer == ground_truth?
  "raw": "...",                 # deepseek 完整 raw 输出
}

并发：用 ThreadPoolExecutor 控制并发，默认 8 路。
速率：单请求 2-3 秒，8 并发 → ~3-4 分钟/100 条 → 8551 条约 30-50 分钟。

用法：
  .venv-cola/bin/python my_try/scripts/distill_cot_data.py \\
      --concurrency 8 --start 0 --end 8551
  # 支持断点续跑（看已有 raw.jsonl 的最后 idx 决定 --start）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd
import requests
from tqdm import tqdm


# ---- 配置 ----
API_KEY_ENV = "CODEBUDDY_API_KEY"
API_KEY_FALLBACK = "ck_fhl595pwk64g.qltQ6Gk3UcvvJvoUDQJitm0VXfLcY-62u9rFc0ZEmsU"
ENDPOINT = "https://copilot.tencent.com/v2/chat/completions"
MODEL = "deepseek-v3-0324"   # Step 4 实测 MCC=0.7268，CoLA 上最准

# 蒸馏 prompt：先要分析过程，最后一行单词答案
DISTILL_PROMPT = """You are a linguistics expert specializing in English grammar.

For the sentence below, briefly analyze its grammar in 2-4 sentences. Consider:
- Subject-verb agreement
- Argument structure (transitivity, complement selection)
- Word order
- Idiomaticity

Then, on the LAST LINE, output ONLY ONE word: "acceptable" or "unacceptable".

Sentence: {sentence}

Analysis:"""

DATA_PATH = Path("cola_data/in_domain_train.tsv")
OUT_DIR = Path("my_try/data")


# ---- API 调用 ----
def chat_distill(sentence: str, api_key: str, max_tokens: int = 512,
                 timeout: int = 60, max_retries: int = 3) -> dict:
    """单条蒸馏请求。返回 {raw, reasoning, answer, error}."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": DISTILL_PROMPT.format(sentence=sentence)}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,   # 确定性输出便于解析
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(ENDPOINT, headers=headers, json=payload,
                                 timeout=timeout, stream=True)
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    return {"raw": "", "reasoning": "", "answer": None, "error": last_err}
                time.sleep(1.5 * (attempt + 1))
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
                if choices:
                    delta = choices[0].get("delta") or {}
                    c = delta.get("content") or ""
                    if c:
                        content += c

            if not content:
                last_err = "empty response"
                time.sleep(1 * (attempt + 1))
                continue

            # 解析：reasoning + 末尾的 acceptable/unacceptable
            answer = parse_last_word(content)
            reasoning = strip_last_answer_line(content)
            return {"raw": content, "reasoning": reasoning, "answer": answer, "error": None}

        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
    return {"raw": "", "reasoning": "", "answer": None, "error": last_err}


def parse_last_word(text: str) -> str | None:
    """从 raw 输出末尾找 acceptable/unacceptable"""
    t = text.strip().lower()
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if not lines:
        return None
    tail = lines[-1]
    # 先判更长的
    if "unacceptable" in tail:
        return "unacceptable"
    if "acceptable" in tail:
        return "acceptable"
    # 兜底：整段查
    if "unacceptable" in t:
        return "unacceptable"
    if "acceptable" in t:
        return "acceptable"
    return None


def strip_last_answer_line(text: str) -> str:
    """去掉末尾仅有 acceptable/unacceptable 的那一行，保留分析"""
    lines = text.rstrip().splitlines()
    if not lines:
        return text
    last = lines[-1].strip().lower()
    # 末行如果就是 acceptable/unacceptable（独立的），去掉
    if last in ("acceptable", "unacceptable"):
        return "\n".join(lines[:-1]).rstrip()
    # 末行如果包含 acceptable/unacceptable（可能带标点）但很短，去掉
    if len(lines[-1].strip()) <= 30 and ("acceptable" in last or "unacceptable" in last):
        return "\n".join(lines[:-1]).rstrip()
    return text.strip()


# ---- 并发 + 断点续跑 ----
COLS = ["source", "label", "first_label", "text"]


def load_existing_indices(out_path: Path) -> set[int]:
    """读已蒸馏的索引（断点续跑）"""
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "idx" in obj:
                    done.add(int(obj["idx"]))
            except json.JSONDecodeError:
                pass
    return done


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DATA_PATH))
    p.add_argument("--outdir", default=str(OUT_DIR))
    p.add_argument("--concurrency", type=int, default=8,
                   help="并发线程数（CodeBuddy API 默认上限不严，8 路较稳）")
    p.add_argument("--max_tokens", type=int, default=512,
                   help="DeepSeek 单次响应 token 上限")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=0,
                   help="0 = 全部")
    args = p.parse_args()

    api_key = os.environ.get(API_KEY_ENV, API_KEY_FALLBACK)

    df = pd.read_csv(args.data, sep="\t", header=None, names=COLS,
                     dtype={"source": str, "label": int, "first_label": str, "text": str},
                     keep_default_na=False)
    end = args.end if args.end > 0 else len(df)
    df = df.iloc[args.start:end].reset_index(drop=False)  # 保留原 index 到 'index' 列

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    raw_out = Path(args.outdir) / "cola_train_cot_raw.jsonl"

    # 断点续跑：跳过已蒸馏的 idx
    done_idx = load_existing_indices(raw_out)
    print(f"📊 训练集总条数: {end - args.start}")
    print(f"   已蒸馏: {len(done_idx)} 条（断点续跑会跳过）")
    todo = [(int(row.index), row.text, int(row.label))
            for row in df.itertuples(index=False)
            if int(row.index) not in done_idx]
    print(f"   待蒸馏: {len(todo)} 条")
    print(f"   并发: {args.concurrency} 路；模型: {MODEL}")

    if not todo:
        print("✅ 已全部完成。")
        return 0

    # 先做连通性测试
    print(f"\n🔎 连通性测试（用第一条样本）...")
    test_idx, test_sent, test_label = todo[0]
    t0 = time.time()
    test = chat_distill(test_sent, api_key, max_tokens=args.max_tokens)
    print(f"   耗时 {time.time() - t0:.1f}s; answer={test['answer']!r}")
    if test["error"]:
        print(f"❌ 测试失败: {test['error']}")
        return 1

    # 主循环（追加写）
    file_lock = Lock()
    f_out = raw_out.open("a", encoding="utf-8")
    bar = tqdm(total=len(todo), desc="distill", dynamic_ncols=True)
    success_count, fail_count = 0, 0
    match_count = 0

    def worker(item):
        idx, sentence, label = item
        gt = "acceptable" if label == 1 else "unacceptable"
        result = chat_distill(sentence, api_key, max_tokens=args.max_tokens)
        record = {
            "idx": idx,
            "sentence": sentence,
            "label": label,
            "ground_truth": gt,
            "answer": result["answer"],
            "match": result["answer"] == gt if result["answer"] else False,
            "reasoning": result["reasoning"],
            "raw": result["raw"],
            "error": result["error"],
        }
        return record

    start_time = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(worker, item) for item in todo]
            for fut in as_completed(futures):
                rec = fut.result()
                with file_lock:
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f_out.flush()
                if rec["error"]:
                    fail_count += 1
                else:
                    success_count += 1
                    if rec["match"]:
                        match_count += 1
                bar.update(1)
                bar.set_postfix(ok=success_count, fail=fail_count, match=match_count)
    finally:
        f_out.close()
        bar.close()

    elapsed = time.time() - start_time
    print(f"\n✅ 蒸馏完成。耗时 {elapsed:.0f}s = {elapsed/60:.1f} min")
    print(f"   成功: {success_count}/{len(todo)}（{success_count/len(todo)*100:.1f}%）")
    print(f"   答对 GT: {match_count}/{len(todo)}（{match_count/len(todo)*100:.1f}%）")
    print(f"   失败: {fail_count}")
    print(f"   原始结果: {raw_out}")

    # 现在过滤出 match=True 的样本
    filtered_path = Path(args.outdir) / "cola_train_cot_filtered.jsonl"
    print(f"\n🔍 过滤 match=True 的样本 → {filtered_path}")
    n_kept = 0
    with raw_out.open("r", encoding="utf-8") as fin, filtered_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("match") and obj.get("reasoning"):
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_kept += 1
    print(f"   保留 {n_kept} 条有效 CoT 样本")
    return 0


if __name__ == "__main__":
    sys.exit(main())
