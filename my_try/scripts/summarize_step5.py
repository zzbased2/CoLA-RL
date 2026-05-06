"""
汇总 Step 5 所有实验结果（baseline / LoRA sweep / QLoRA 4B / full SFT 等），
输出一张 markdown 表格 + 全景图。
"""
import json
from pathlib import Path

RESULT_DIRS = [
    "my_try/results/step5_lora",
    "my_try/results/step5_fullsft",
]

rows = []
for d in RESULT_DIRS:
    for f in sorted(Path(d).glob("res_*.json")):
        try:
            data = json.loads(f.read_text())
            r = data["results"][0] if "results" in data else data
            # 归一化字段
            name = f.stem.replace("res_lora_", "").replace("res_fullsft_", "")
            rows.append({
                "name": name,
                "mcc": r.get("mcc", 0),
                "acc": r.get("accuracy", 0),
                "parse_fail": r.get("parse_fail", 0),
                "avg_tokens": r.get("avg_new_tokens", 0),
                "file": str(f),
            })
        except Exception as e:
            print(f"⚠️  skip {f}: {e}")

rows.sort(key=lambda x: -x["mcc"])
print(f"\n{'排名':>3} | {'实验':50s} | {'MCC':>7s} | {'Acc':>7s} | {'parse_fail':>4s} | avg_tok")
print("-" * 110)
for i, r in enumerate(rows, 1):
    star = " ⭐" if r["mcc"] > 0.64 else ""
    print(f"{i:3d} | {r['name']:50s} | {r['mcc']:.4f} | {r['acc']:.4f} | {r['parse_fail']:4d} | {r['avg_tokens']:6.1f}{star}")
