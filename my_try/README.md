# my_try：CoLA-RL 本地尝试（14G 显存版）

> 这个目录存放本次本地复现尝试的**所有产出**（计划文档、脚本、评测结果、总结报告），
> 原项目的 `verl/`、`recipe/`、`scripts/`、`cola_data/` 等**保持原样不动**。

## 🌐 Git 仓库信息

本地 fork 自原项目并在此基础上扩展。采用标准 **fork 工作流**：

| remote | URL | 用途 |
| --- | --- | --- |
| **origin** | <https://github.com/zzbased2/CoLA-RL.git> | **我的 fork**，本次所有工作的推送目标 |
| **upstream** | <https://github.com/ytzfhqs/CoLA-RL.git> | 原作者仓库，用于同步上游更新 |

**日常推送**（分支跟踪已建立）：

```bash
cd /data/workspace/Github-open/CoLA-RL
git add my_try/
git commit -m "your message"
git push   # 自动推到 origin/main
```

**同步上游原作者的更新**（如有新版本）：

```bash
git fetch upstream
git merge upstream/main       # 或 git rebase upstream/main
git push
```

---

## 🗺 文档导航

**新手从这里入门**：

- 🎯 **总计划与背景**：[`my_try.md`](./my_try.md)
- 📘 **教程（独立可读）**：[`lora_sft_tutorial.md`](./lora_sft_tutorial.md) ← **想做 LoRA SFT 看这个**

### 📘 各阶段汇总报告（`docs/`）

| 文件 | 主题 |
| --- | --- |
| [`docs/environment.md`](./docs/environment.md) | 环境搭建记录（Python/torch/transformers 版本、踩坑点） |
| [`docs/baseline_summary.md`](./docs/baseline_summary.md) | **Step 3** 汇总：4 模型 zero-shot MCC |
| [`docs/step4_summary.md`](./docs/step4_summary.md) | **Step 4** 汇总：16 组 prompt 变体 + 5 经典 ML + 4 个 API 大模型 |
| [`docs/step5_summary.md`](./docs/step5_summary.md) | **Step 5** 汇总：LoRA SFT (E1/E4) + 全参 SFT (E1-FULL v1/v2) |
| [`docs/step6_summary.md`](./docs/step6_summary.md) | **Step 6** 汇总：LoRA-GRPO（mode collapse 失败案例研究）|

### 🐍 脚本（`scripts/`）

| 文件 | 类别 | 说明 |
| --- | :-: | --- |
| [`scripts/sanity_check.py`](./scripts/sanity_check.py) | 环境 | CUDA + 模型加载验证 |
| [`scripts/prepare_sft_data.py`](./scripts/prepare_sft_data.py) | 数据 | CoLA TSV → SFT messages JSONL |
| [`scripts/eval_baseline.py`](./scripts/eval_baseline.py) | 评测 | LLM zero-shot 评测（plain/fewshot/cot/thinking）|
| [`scripts/eval_classical.py`](./scripts/eval_classical.py) | 评测 | 经典 ML baseline (Majority/Random/LR) |
| [`scripts/eval_api_models.py`](./scripts/eval_api_models.py) | 评测 | CodeBuddy API 大模型评测 |
| [`scripts/train_lora_sft.py`](./scripts/train_lora_sft.py) | 训练 | **LoRA SFT 主力脚本** ⭐ |
| [`scripts/eval_lora.py`](./scripts/eval_lora.py) | 评测 | LoRA adapter 评测（含 GRPO 后的 adapter）|
| [`scripts/train_full_sft.py`](./scripts/train_full_sft.py) | 训练 | 全参 SFT |
| [`scripts/eval_fullsft.py`](./scripts/eval_fullsft.py) | 评测 | 全参 SFT 模型评测 |
| [`scripts/train_lora_grpo.py`](./scripts/train_lora_grpo.py) | 训练 | LoRA-GRPO（trl GRPOTrainer 纯 PyTorch rollout）|

### 📊 数据（`data/`）

| 文件 | 条数 | 用途 |
| --- | :-: | --- |
| `data/sft_train.jsonl` | 8551 | SFT 训练（messages 格式）|
| `data/sft_val.jsonl` | 527 | SFT 验证 |

### 📈 实验结果（`results/`，按 step 分组）

```
results/
├── step3_baseline/        Step 3 zero-shot baseline
│   ├── baseline_results.json     4 模型的 plain MCC
│   └── baseline_classical.json   5 个经典 ML baseline
├── step4_variants/        Step 4：12 个 prompt 变体
│   └── res_{model}_{variant}.json × 12  (4 模型 × 3 变体)
├── step4_api/             Step 4：4 个 API 大模型
│   └── res_api_{model}.json × 4
├── step5_lora/            Step 5：LoRA SFT (2 个)
│   ├── res_lora_Qwen3-0.6B-lora-E1-r16.json
│   └── res_lora_Qwen3-1.7B-lora-E4-r16.json
├── step5_fullsft/         Step 5：全参 SFT (2 个)
│   ├── res_fullsft_Qwen3-0.6B-fullsft-E1-FULL.json    (v1, lr 太保守)
│   └── res_fullsft_Qwen3-0.6B-fullsft-E1-FULL-v2.json (v2, 激进版)
└── step6_grpo/            Step 6：LoRA-GRPO (3 个，全部失败案例)
    ├── res_lora_*-grpo-E1-GRPO-r16.json     (v1, ±1 reward)
    ├── res_lora_*-grpo-E1-GRPO-v2-r16.json  (v2, balanced reward)
    └── res_lora_*-grpo-E1-GRPO-v3-r16.json  (v3, balanced + KL + 半量)
```

### 📦 依赖清单（`requirements/`）

| 文件 | 说明 |
| --- | --- |
| [`requirements/requirements-local.in`](./requirements/requirements-local.in) | 源依赖清单（15 个包，带下限） |
| [`requirements/requirements-local.txt`](./requirements/requirements-local.txt) | `pip freeze` 产出（166 个精确版本） |

---

## 🚀 如何运行本目录的脚本

**所有脚本都从项目根目录执行**（不是从 `my_try/` 目录执行），因为脚本里用的是项目根的相对路径（如 `cola_data/...`、`model/...`、`my_try/...`）：

```bash
cd /data/workspace/Github-open/CoLA-RL
source .venv-cola/bin/activate

# 环境验证
python my_try/scripts/sanity_check.py

# Step 3: 跑 LLM zero-shot baseline
python my_try/scripts/eval_baseline.py --prompt plain

# Step 3: 跑经典 ML baseline
python my_try/scripts/eval_classical.py

# Step 4: 各 prompt 变体（指定 --out 保存到 step4_variants/）
python my_try/scripts/eval_baseline.py \
    --model model/Qwen3-0.6B --prompt fewshot \
    --out my_try/results/step4_variants/res_qwen3-0.6b_fewshot.json

# Step 5: LoRA SFT
python my_try/scripts/prepare_sft_data.py    # 准备数据（仅一次）
nohup python my_try/scripts/train_lora_sft.py \
    --model model/Qwen3-0.6B --exp E1 --rank 16 --epochs 3 \
    --bsz 8 --grad_accum 4 --lr 2e-4 --max_length 256 &

# Step 5: 评测 LoRA
python my_try/scripts/eval_lora.py \
    --adapter my_try/ckpt/Qwen3-0.6B-lora-E1-r16
```

---

## 📁 完整目录树

```
my_try/                              ← 你在这里（本次所有产出根目录）
├── README.md                        本文件（导航）
├── my_try.md                        总计划
├── lora_sft_tutorial.md             ⭐ LoRA SFT 教程（独立可读）
│
├── docs/                            📘 各 step 汇总报告
│   ├── environment.md
│   ├── baseline_summary.md          (Step 3)
│   ├── step4_summary.md
│   ├── step5_summary.md
│   └── step6_summary.md
│
├── scripts/                         🐍 所有 Python 脚本（10 个）
│   ├── sanity_check.py              环境验证
│   ├── prepare_sft_data.py          数据准备
│   ├── eval_baseline.py             LLM 评测
│   ├── eval_classical.py            经典 ML baseline
│   ├── eval_api_models.py           API 大模型评测
│   ├── train_lora_sft.py            ⭐ LoRA SFT
│   ├── eval_lora.py                 LoRA adapter 评测
│   ├── train_full_sft.py            全参 SFT
│   ├── eval_fullsft.py              全参 SFT 评测
│   └── train_lora_grpo.py           LoRA-GRPO
│
├── data/                            📊 SFT 数据（messages JSONL）
│   ├── sft_train.jsonl              8551 条
│   └── sft_val.jsonl                527 条
│
├── results/                         📈 实验结果（按 step 分组）
│   ├── step3_baseline/              4 + 5 baselines
│   ├── step4_variants/              12 prompt variants
│   ├── step4_api/                   4 API models
│   ├── step5_lora/                  2 LoRA SFT
│   ├── step5_fullsft/               2 full SFT
│   └── step6_grpo/                  3 GRPO（含失败案例）
│
├── requirements/                    📦 依赖清单
│   ├── requirements-local.in
│   └── requirements-local.txt
│
└── ckpt/                            💾 训练 checkpoint (.gitignore 已忽略)
    ├── Qwen3-0.6B-lora-E1-r16/      ⭐ Step 5 LoRA SFT
    ├── Qwen3-1.7B-lora-E4-r16/
    ├── Qwen3-0.6B-fullsft-E1-FULL-v2/
    ├── Qwen3-0.6B-grpo-E1-GRPO-r16/      v1
    ├── Qwen3-0.6B-grpo-E1-GRPO-v2-r16/   v2
    └── Qwen3-0.6B-grpo-E1-GRPO-v3-r16/   v3
```

外层（项目根）：

```
CoLA-RL/
├── .gitignore                       忽略 .venv-cola/ 和 model/Qwen*/
├── .venv-cola/                      虚拟环境（已忽略）
├── model/                           模型权重（已忽略）
│   └── Qwen3-0.6B / 1.7B / 3.5-0.8B / 3.5-2B
├── cola_data/                       原始 CoLA 数据（原项目，未动）
├── my_try/                          ← 本目录
└── (原项目 8 个文件：README/LICENSE/train_cls.py/...)
```

---

## 🏆 关键结果速览

| 实验 | 方法 | MCC | 备注 |
| --- | --- | :-: | --- |
| Qwen3-0.6B baseline | zero-shot | 0.000 | 起点 |
| 经典 LR baseline | TF-IDF + LR | 0.176 | 上限 |
| **E1 LoRA SFT** | **0.6B + LoRA r=16** | **0.5406** | ⭐ Step 5 主力 |
| **E4 LoRA SFT** | **1.7B + LoRA r=16** | **0.6246** | ⭐ Step 5 |
| E1-FULL v2 全参 SFT | 0.6B 全参 lr=5e-5 | 0.5361 | LoRA 略胜全参 |
| Step 6 GRPO（v1/v2/v3）| 3 种配置 | **全部 0.0000** | mode collapse |
| API 锚点（deepseek-v3）| zero-shot | 0.7268 | 能力天花板 |

详细分析见 [`docs/step5_summary.md`](./docs/step5_summary.md) 和 [`docs/step6_summary.md`](./docs/step6_summary.md)。
