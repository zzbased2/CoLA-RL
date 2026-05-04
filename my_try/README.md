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

**首次配置该 fork 工作流的历史步骤**（已完成，仅留作记录）：

```bash
# 1. 原 origin 指向作者仓库，把它重命名为 upstream
git remote rename origin upstream

# 2. 新增 origin 指向自己的 fork
git remote add origin https://github.com/zzbased2/CoLA-RL.git

# 3. 推送并建立分支跟踪
git push -u origin main
```

---

## 🗺 导航

**总计划文档**：[`my_try.md`](./my_try.md) ← **从这里看起**

### 📘 文档类（Markdown）

| 文件 | 说明 |
| --- | --- |
| [`my_try.md`](./my_try.md) | **总计划**：Step 1-6 的完整流程、实验矩阵、状态追踪 |
| [`environment.md`](./environment.md) | 环境搭建记录（Python/torch/transformers 版本、踩坑点） |
| [`baseline_summary.md`](./baseline_summary.md) | Step 3 汇总：4 模型 zero-shot MCC |
| [`step4_summary.md`](./step4_summary.md) | Step 4 汇总：16 组实验（4 模型 × 4 变体 + 5 经典 ML） |

### 🐍 脚本类（Python）

| 文件 | 说明 |
| --- | --- |
| [`sanity_check.py`](./sanity_check.py) | 环境验证（CUDA + 模型加载） |
| [`eval_baseline.py`](./eval_baseline.py) | LLM 评测脚本（plain/fewshot/cot/thinking） |
| [`eval_classical.py`](./eval_classical.py) | 经典 ML baseline（Majority/Random/LR） |

### 📊 结果类（JSON）

| 文件 | 说明 |
| --- | --- |
| `baseline_results.json` | Step 3：4 模型 zero-shot 结果 |
| `baseline_classical.json` | 5 个经典 ML baseline 结果 |
| `res_{model}_{variant}.json` × 12 | Step 4：4 模型 × 3 变体（fewshot/cot/thinking） |

### 📦 依赖清单

| 文件 | 说明 |
| --- | --- |
| [`requirements-local.in`](./requirements-local.in) | 源依赖清单（15 个包，带下限） |
| [`requirements-local.txt`](./requirements-local.txt) | `pip freeze` 产出（166 个精确版本） |

## 🚀 如何运行本目录的脚本

**所有脚本都从项目根目录执行**（不是从 `my_try/` 目录执行），因为脚本里用的是项目根的相对路径：

```bash
cd /data/workspace/Github-open/CoLA-RL
source .venv-cola/bin/activate

# 环境验证
python my_try/sanity_check.py

# 跑 LLM baseline（Step 3）
python my_try/eval_baseline.py --prompt plain

# 跑经典 ML baseline
python my_try/eval_classical.py

# 跑 Step 4 四个变体
python my_try/eval_baseline.py --model model/Qwen3-0.6B --prompt fewshot
python my_try/eval_baseline.py --model model/Qwen3-0.6B --prompt cot
python my_try/eval_baseline.py --model model/Qwen3-0.6B --prompt thinking
```

## 📁 目录树

```
CoLA-RL/
├── .gitignore                   # ★ 忽略 .venv-cola/ 和 model/Qwen*/
├── my_try/                      # ← 你在这里，本次所有产出都在这
│   ├── README.md                # 本文件
│   ├── my_try.md                # 总计划
│   ├── environment.md           # 环境记录
│   ├── baseline_summary.md      # Step 3 总结
│   ├── step4_summary.md         # Step 4 总结
│   ├── sanity_check.py
│   ├── eval_baseline.py
│   ├── eval_classical.py
│   ├── requirements-local.in
│   ├── requirements-local.txt
│   ├── baseline_results.json
│   ├── baseline_classical.json
│   └── res_*.json × 12
│
├── model/                       # 模型权重（.gitignore 已忽略）
│   ├── Qwen3-0.6B/
│   ├── Qwen3-1.7B/
│   ├── Qwen3.5-0.8B/
│   └── Qwen3.5-2B/
│
├── .venv-cola/                  # 虚拟环境（.gitignore 已忽略）
│
└── (原项目 8 个文件：README/LICENSE/train_cls.py/...)  # 保持原状
```
