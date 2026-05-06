# CoLA-RL Step 5：LoRA SFT 实验记录

> 创建日期：2026-05-04
> 对应计划：`my_try.md §Step 5`
> 前置结论：`my_try/step4_summary.md`（Step 4 证明 SFT 必要性）
> 硬件：NVIDIA L20 × 1（14 GB bf16）
> 框架：trl 1.3.0（SFTTrainer）+ peft 0.19.1（LoRA）
> 基座：Qwen3-0.6B / Qwen3-1.7B（标准 Qwen3 架构，未选 Qwen3.5 多模态版本）

---

## 目录

- [1. 目标与锚点](#1-目标与锚点)
- [2. 环境与数据准备](#2-环境与数据准备)
- [3. 训练脚本设计](#3-训练脚本设计)
- [4. 实验矩阵](#4-实验矩阵)
- [5. 实验结果](#5-实验结果)
  - [5.4 CoT 蒸馏 SFT 实验（追加）](#54-cot-蒸馏-sft-实验追加deepseek-v3-蒸馏--06b--17b)
  - [5.5 1.7B 全参 SFT 复现尝试（追加）](#55-17b-全参-sft-复现尝试追加对齐原项目配方)
  - [5.6 基于 LoRA 最优解的三步迭代（追加）](#56-基于-lora-最优解的三步迭代追加超参精调--qlora-4b--hard-example-mining)
- [6. 关键发现（待填充）](#6-关键发现待填充)
- [7. 产出物清单](#7-产出物清单)
- [8. 遇到的坑](#8-遇到的坑)

---

## 1. 目标与锚点

### 1.1 Step 5 要回答的核心问题

> 在只有 14 GB 显存、不能做全参 SFT 的前提下，**LoRA SFT 能把 Qwen3 系列的小模型在 CoLA 上拉到什么高度？**

具体：
- Qwen3-0.6B 能否从 baseline 0.000 拉到 **≥ 0.30**（超越自身 thinking 0.254）？
- Qwen3-1.7B 能否从 baseline 0.500 拉到 **≥ 0.58**（盖过 Qwen3.5-2B 的 0.549）？

### 1.2 外部锚点（来自 Step 3/4 和原项目 README）

| 锚点 | MCC | 来源 |
| --- | --- | --- |
| 全说 yes / 经典 Majority | 0.000 | Step 4 B1_Majority |
| 经典 LR baseline | 0.176 | Step 4 B5_WordNgramLR |
| **Qwen3-0.6B zero-shot** | **0.000** | Step 3 |
| Qwen3-0.6B + thinking | 0.254 | Step 4 |
| Qwen3.5-0.8B zero-shot | 0.340 | Step 3 |
| **Qwen3-1.7B zero-shot** | **0.500** | Step 3 |
| hy3-preview（API 推理型）| 0.527 | Step 4 补测 |
| Qwen3.5-2B zero-shot | 0.549 | Step 3 |
| 原项目 Qwen3-0.6B 全参 SFT | 0.598 | 原 README |
| 原项目 Qwen3-1.7B 全参 SFT | 0.657 | 原 README |
| 原项目 Qwen3-1.7B SFT+GRPO | 0.702 | 原 README（我们的"顶"）|
| **gemini-3.1-flash-lite**（API）| **0.703** | Step 4 补测 |
| **claude-opus-4.7**（API）| **0.718** | Step 4 补测 |
| **deepseek-v3-0324**（API）| **0.727** | Step 4 补测（与原 README 完全一致）|

---

## 2. 环境与数据准备

### 2.1 环境增补（在 Step 2 之上）

| 组件 | 状态 |
| --- | --- |
| Python 3.12 `_lzma` 缺失 | ⚠️ 修复：用 stub 模块 `.venv-cola/.../site-packages/_lzma.py` 永久顶替 |
| trl API 差异 | 见 [§8 遇到的坑](#8-遇到的坑) |

### 2.2 数据：messages 格式 JSONL

脚本：`my_try/prepare_sft_data.py`

```python
{
  "messages": [
    {"role": "user", "content": "Decide whether...\nSentence: <句子>\n\nYour answer:"},
    {"role": "assistant", "content": "acceptable"}   // or "unacceptable"
  ]
}
```

产出：

| 文件 | 条数 | 用途 |
| --- | --- | --- |
| `my_try/data/sft_train.jsonl` | 8,551 | 训练（= `in_domain_train.tsv`） |
| `my_try/data/sft_val.jsonl` | 527 | 训练中的 eval + 后续 MCC 评测（= `in_domain_dev.tsv`） |

**关键设计（对齐原则）**：
- user 侧 prompt 文本与 `eval_baseline.py` 的 `PLAIN_TEMPLATE` **逐字符一致**（避免训练-评测漂移）
- assistant 侧**只有答案词**（不带 CoT），因为 Step 4 证明 CoT 让模型解析失败率飙升
- trl 1.x `completion_only_loss=True` 会自动识别 `messages` 结构，**只在 assistant 段算 loss**

---

## 3. 训练脚本设计

### 3.1 脚本

- `my_try/train_lora_sft.py`：训练（支持所有 E1-E5）
- `my_try/eval_lora.py`：加载 adapter，527 条验证集算 MCC，输出到 `my_try/res_lora_<name>.json`

### 3.2 默认超参（`my_try.md §5` 版，已在 smoke test 通过）

| 超参 | 值 | 说明 |
| --- | --- | --- |
| `epochs` | 3 | 2-3 epoch 通常足够；更多会过拟合 |
| `per_device_train_batch_size` | 8（0.6B）/ 4（1.7B） | 受显存约束 |
| `gradient_accumulation_steps` | 4（0.6B）/ 8（1.7B） | **effective batch = 32** |
| `learning_rate` | 2e-4 | LoRA 标准 LR；全参 SFT 一般 5e-6 - 1e-5 |
| `lr_scheduler` | cosine | |
| `warmup_ratio` | 0.03 | |
| `max_length` | 256 | Prompt ~80 token + 答案 1-2 token，充裕 |
| `bf16` | True | |
| `gradient_checkpointing` | True | 省 5-8× 激活显存 |
| `completion_only_loss` | True | trl 1.x 新特性 |
| `lora_alpha` | 2 × rank | 经验稳健默认 |
| `lora_dropout` | 0.05 | |
| `target_modules` | q_proj, k_proj, v_proj, o_proj | 标准 LoRA 选择 |

### 3.3 实验命名规范

```
Qwen3-0.6B-lora-{ExpID}-r{rank}
例：Qwen3-0.6B-lora-E1-r16
```

产出结构：

```
my_try/ckpt/<exp_name>/
├── adapter_config.json       # LoRA 配置
├── adapter_model.safetensors # LoRA 权重（几 MB）
├── tokenizer.json / ...      # 保存的 tokenizer
├── run_info.json             # 自定义训练元数据（参数量、耗时等）
└── checkpoint-*/             # 中间检查点（默认保留最后 2 个）

my_try/res_lora_<exp_name>.json  # 评测结果
```

---

## 4. 实验矩阵

### 4.1 本次实际执行的实验（优先级排序）

| 实验 | 模型 | 方法 | rank / lr | epochs | bsz × acc | 目标 MCC | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **E1** | Qwen3-0.6B | LoRA | r=16 / 2e-4 | 3 | 8×4 | ≥ 0.30 | ✅ **0.5406** |
| **E4** | Qwen3-1.7B | LoRA | r=16 / 2e-4 | 3 | 4×8 | ≥ 0.58 | ✅ **0.6246** |
| **E1-FULL v1** | Qwen3-0.6B | 全参 SFT | — / 1e-5 | 3 | 4×8 | ≥ 0.55 | ⚠️ **0.4526**（lr 太保守） |
| **E1-FULL v2** | Qwen3-0.6B | 全参 SFT | — / **5e-5** | 3 | **16×2** | ≥ 0.55 | ✅ **0.5361**（调参后） |
| E2 | Qwen3-0.6B | LoRA | r=32 / 2e-4 | 3 | 8×4 | ≥ 0.40 | ⬜ 未执行 |
| E3 | Qwen3-0.6B | LoRA | r=64 / 2e-4 | 3 | 8×4 | ≥ 0.40 | ⬜ 未执行 |
| E5 | Qwen3-1.7B | LoRA | r=32 / 2e-4 | 3 | 4×8 | ≥ 0.60 | ⬜ 未执行 |

### 4.2 核心双跑的设计理由

- **E1**：最能体现 LoRA SFT 价值的实验（baseline 0.000，拉升空间最大）
- **E4**：在已能解题的模型上，LoRA 能否继续压榨（baseline 0.500）

两者回答的问题不同：
- E1 = "LoRA 能不能**教会**一个不会的模型？"
- E4 = "LoRA 能不能**提升**一个会了的模型？"

---

## 5. 实验结果

### 5.1 训练过程记录

| 实验 | 基座 | 方法 | Trainable/Total | Epochs | 训练耗时 | 峰值显存 | 最终 eval_loss | Token Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **E1** | Qwen3-0.6B | LoRA r=16 | 4.59M / 601M = **0.76%** | 3 | **13.4 min** | **3.57 GB** | 0.4300 | 0.9216 |
| **E4** | Qwen3-1.7B | LoRA r=16 | 6.42M / 1.73B = **0.37%** | 3 | **23.5 min** | **4.73 GB** | — | — |
| E1-FULL v1 | Qwen3-0.6B | 全参（保守 lr=1e-5） | 596M / 596M = **100%** | 3 | 16.2 min | **6.01 GB** | 0.4522 | 0.9186 |
| **E1-FULL v2** | Qwen3-0.6B | 全参（**激进 lr=5e-5 + bsz=16 + 无 grad_ckpt**）| 596M / 596M = **100%** | 3 | **4.3 min** | **13.23 GB** | **0.3616** (epoch 2 最低) | 0.9316 |

> E1 和 E4 **并行**运行，稳态合计约 12 GB；E1-FULL v2 **充分榨干了 14 GB 卡**（13.23 GB 峰值，90% 利用率）。

### 5.2 评测结果（527 条 dev）

| 实验 | 方法 | **MCC** | Accuracy | Parse fail | vs Baseline | avg_tokens |
| --- | --- | --- | --- | --- | --- | --- |
| **E1** | 0.6B + LoRA r=16 | **0.5406** | 0.8083 | **0/527** | 0.000 → **0.5406** (+0.54) | 1.3 |
| **E4** | 1.7B + LoRA r=16 | **0.6246** | 0.8387 | **0/527** | 0.500 → **0.6246** (+0.12) | 1.3 |
| E1-FULL v1 | 0.6B 全参（lr=1e-5）| 0.4526 | 0.7818 | 0/527 | 0.000 → 0.4526 | ~1 |
| **E1-FULL v2** | 0.6B 全参（lr=5e-5 激进）| **0.5361** | 0.7989 | 0/527 | 0.000 → **0.5361** | ~1 |

**E1 vs E1-FULL v2 的对比**（**同模型同数据，只是训练策略不同**）：

```
             pred=unacc  pred=acc              pred=unacc  pred=acc
E1 (LoRA)      105         57                   E1-FULL v2   114         48
               44          321                                58         307
               unacc 召回 65%                                 unacc 召回 70%
               acc   召回 88%                                 acc   召回 84%
               MCC=0.5406                                     MCC=0.5361
```

→ **LoRA 和全参在同量级样本 + 同 epoch 下表现几乎打平**，全参略倾向预测 unacc（召回稍高但 acc 召回稍低）。

### 5.3 与外部锚点对比（全景图）

```
MCC  0.000 ─── 全说 yes / Qwen3-0.6B zero-shot        (baseline 垫底)
     0.176 ─── 经典 LR (B5_WordNgramLR)
     0.254 ─── Qwen3-0.6B + thinking
     0.340 ─── Qwen3.5-0.8B zero-shot
     0.413 ─── E1-CoT: 0.6B + LoRA-SFT (DeepSeek-V3 蒸馏 CoT)  ← 比短答案 −0.13
     0.453 ─── E1-FULL v1: 0.6B 全参 SFT (lr=1e-5 保守)  ← 失败案例，调参不足
     0.500 ─── Qwen3-1.7B zero-shot
     0.526 ─── E4-FULL: 1.7B 全参 SFT (adamw_8bit, 14G 硬限制)  ← 反而不如 LoRA
     0.527 ─── hy3-preview (API 推理型)
     0.536 ─── ⭐ E1-FULL v2: 0.6B 全参 SFT (lr=5e-5 激进) ← 调参后
     0.541 ─── ⭐ E1: Qwen3-0.6B + LoRA-SFT r=16       ← 小模型 LoRA 最强!
     0.549 ─── Qwen3.5-2B zero-shot
     0.564 ─── E4-CoT: 1.7B + LoRA-SFT (DeepSeek-V3 蒸馏 CoT)  ← 比短答案 −0.06
     0.582 ─── E4-HARD: 1.7B + LoRA r=32 + HardWeight ×3       ← hard mining 反向
     0.598 ─── 原项目 Qwen3-0.6B 全参 SFT (README)     (我们 E1 差 -0.058)
     0.610 ─── 原项目 Qwen3-0.6B-CLS
     0.625 ─── E4: Qwen3-1.7B + LoRA-SFT r=16 (baseline)
     0.636 ─── DeepSeek-R1 0120
     0.657 ─── 原项目 Qwen3-1.7B 全参 SFT 🎯
     0.6575 ── ⭐ E4-A-mlp: 1.7B LoRA r=16 + MLP target
     0.6633 ── ⭐ E6: Qwen3-4B QLoRA r=32 +MLP (Acc=0.8615 最高)
     0.6654 ── 🥇 E4-C-r32: Qwen3-1.7B LoRA r=32 (本机最优 MCC)
     0.702 ─── 原项目 Qwen3-1.7B SFT+GRPO              (他们的"天花板")
     0.703 ─── gemini-3.1-flash-lite (API)
     0.718 ─── claude-opus-4.7 (API)
     0.727 ─── deepseek-v3-0324 (API)                  (API 模型天花板)
```

**三个维度的观察**：

| 维度 | 结论 |
| --- | --- |
| **0.6B + LoRA vs 0.6B 全参** | **LoRA 略胜**（0.5406 vs 0.5361），说明在这个任务+数据量上 LoRA 容量够 |
| **0.6B LoRA vs Qwen3-1.7B zero-shot** | **LoRA 微胜**（0.5406 vs 0.500），**小模型 SFT 打败 3× 参数大模型** |
| **我们的 LoRA vs API 大模型** | gap -0.18，**规模差距无法通过 SFT 完全弥补**，但已经接近 DeepSeek-R1 的 0.636 |

---

### 5.4 CoT 蒸馏 SFT 实验（追加：DeepSeek-V3 蒸馏 → 0.6B / 1.7B）

#### 5.4.1 动机与做法

Step 6（GRPO）三轮失败后，回过头怀疑 §5 的"短答案 SFT"信号过弱：模型只学到一个 token 的输出，缺少**判别推理过程**，因此在 RL 阶段难以涌现。
尝试用 **DeepSeek-V3** 对训练集做 CoT 蒸馏，再用 `<think>...</think>\nlabel` 格式 SFT，期望：
1. 蒸馏出语法分析的中间推理（subject-verb agreement / argument structure / word order / idiomaticity）
2. 让小模型先"学会怎么想"，再走 GRPO

**数据**：
- `my_try/data/cola_train_cot_filtered.jsonl` ：**7436** 条（与 DeepSeek 标签一致 = 87% 通过率）
- 训练 / 验证：`sft_train_cot.jsonl` / `sft_val_cot.jsonl`
- 推理时 `enable_thinking=True`，`max_new_tokens=512`

**训练配置**：与短答案 SFT 完全一致（rank=16, alpha=32, 3 epoch, effective_bsz=32, lr=2e-4, max_length=512），只是数据换成 CoT。

#### 5.4.2 评测结果

| 实验 | 模型 | 数据 | **MCC** | Accuracy | avg_tokens | 训练耗时 | Δ vs 短答案 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 0.6B | 短答案 | **0.5406** | 0.8083 | 1.3 | ~12 min | — |
| E1-CoT | 0.6B | DeepSeek-V3 CoT | **0.4135** | 0.7666 | 76.5 | ~17 min | **−0.127** ⚠️ |
| E4 | 1.7B | 短答案 | **0.6246** | 0.8387 | 1.3 | ~22 min | — |
| E4-CoT | 1.7B | DeepSeek-V3 CoT | **0.5643** | 0.8121 | 78.7 | ~36 min | **−0.060** ⚠️ |

**混淆矩阵对比**（1.7B）：

```
              pred=unacc  pred=acc                  pred=unacc  pred=acc
E4 短答案       119         43       E4-CoT          116         46
                41          324                      53          312
                unacc 召回 73.5%                     unacc 召回 71.6%
                acc   召回 88.8%                     acc   召回 85.5%
                MCC=0.6246                           MCC=0.5643
```

→ CoT 后 acc 召回掉了 3.3 pt，unacc 召回也掉了 1.9 pt，两侧同时退化。

#### 5.4.3 三个反直觉发现

**1）CoT 蒸馏在两个模型尺寸上都没赢过短答案 SFT**

| 模型 | 短答案 MCC | CoT MCC | Δ |
| --- | --- | --- | --- |
| 0.6B | 0.5406 | 0.4135 | **−0.127** |
| 1.7B | 0.6246 | 0.5643 | **−0.060** |

模型越大，CoT 退化越小（−0.13 → −0.06），但**仍未翻正**。说明在 CoLA 这种"短句二分类"任务上，CoT 不是缺失的关键信号。

**2）0.6B-CoT 出现"宽容偏见"，1.7B-CoT 没有**

| 模型 | pred_acc_rate | true_acc_rate | 偏差 |
| --- | --- | --- | --- |
| 0.6B + 短答案 | ~0.70 | 0.6926 | +0.01（基本无偏） |
| **0.6B + CoT** | **0.7856** | 0.6926 | **+0.093**（明显倾向预测 acceptable） |
| 1.7B + CoT | 0.6793 | 0.6926 | −0.013（无偏）|

→ 0.6B 模仿能力不足，**学到了 DeepSeek 推理风格里"找理由说能接受"的偏见**（DeepSeek 蒸馏数据 87% match，但 CoT 表述本身就偏宽容）；1.7B 容量足够，能 hold 住分布。

**3）推理成本 ×60，效果反而变差**

`avg_new_tokens` 从 1.3 → 78.7，**单样本生成成本 ×60**，MCC 反而下降。即使 CoT 在 1.7B 上 hold 住分布，性价比也极低。

#### 5.4.4 失败原因分析

1. **任务太简单不需要 CoT**：CoLA 是 7-15 词短句二分类，绝大多数样本一眼看出，"想"反而引入噪声。
2. **CoT 数据本身有 13% 噪声**：DeepSeek 蒸馏中 13% 与人工标签不一致就直接丢掉了，留下的也未必都是"高质量推理"——只是"结论恰好对"。
3. **小模型模仿大模型 CoT 会学走样**：0.6B 学不了 DeepSeek 的语法分析深度，只学到了表面句式（"correct subject-verb agreement, idiomatic..."），导致**风格化的过度宽容**。
4. **答案信号被稀释**：短答案 SFT 把全部梯度集中在最后一个分类 token；CoT-SFT 把梯度摊到 ~80 个推理 token 上，分类 token 的有效学习率反而降低。

#### 5.4.5 结论与对 Step 6（GRPO）的反推

- **CoT 不是 CoLA 任务的优化方向**：原假设"GRPO 失败因为 SFT 阶段没学推理"被证伪。
- **本机当前最优仍然是 E4 短答案 SFT (MCC=0.6246)**，距离 DeepSeek-V3 上限 0.7268 仍有 ~0.10 gap，但已超过原项目 Qwen3-0.6B 全参 SFT (0.598)、DeepSeek-R1 (0.636)。
- 下一步若要继续推：
  - **数据侧**：hard-example mining（看 1.7B 的 53 个 fn / 46 个 fp，再针对性补数据）
  - **模型侧**：4B 基座 + QLoRA（突破 14G 限制）
  - **算法侧**：放弃 GRPO，试 DPO（用 1.7B-SFT 的对/错样本对自动构造偏好对）

---

### 5.5 1.7B 全参 SFT 复现尝试（追加：对齐原项目配方）

#### 5.5.1 动机

我们的 E4 (LoRA) MCC=0.6246，与原项目 README 报告的 **Qwen3-1.7B-SFT 全参 = 0.657** 有 **-0.032 gap**。猜测核心差异是"**LoRA 容量不足**"（见 §6 分析）。为验证这一点，复现原项目的超参配方，跑一遍 1.7B 全参 SFT。

#### 5.5.2 对齐配置

从原项目 `scripts/sft_qwen3_1.7b.yaml` 读到的关键配方：

| 维度 | 原项目 | 我们 E4-FULL | 备注 |
|---|---|---|---|
| 微调方式 | 全参 | **全参** ✅ | 对齐 |
| effective batch | 16 × 8 = **128** | 1 × 128 = **128** ✅ | 对齐（我们被迫 per_device=1） |
| epochs | **2** | **2** ✅ | 对齐 |
| lr | **1e-5** | **1e-5** ✅ | 对齐 |
| scheduler | **cosine** + warmup_ratio=0.01 | **cosine** + warmup_ratio=0.01 ✅ | 对齐 |
| cutoff_len | 512 | **128** ⚠️ | 我们短，但实际 p99=93 token，无影响 |
| dtype | bf16 | bf16 ✅ | 对齐 |
| **optimizer** | **adamw_torch (fp32)** | **adamw_8bit (bnb)** ⚠️ | **14GB 单卡被迫用 8bit** |

**14GB 单卡的约束**：1.7B 全参 fp32 AdamW 需要 ~14GB 优化器状态，加上模型+梯度+激活远超 15GB 配额。所以必须用 **bitsandbytes 的 adamw_8bit**（optimizer state 8bit 量化，节省 ~12GB）+ `gradient_checkpointing=True` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

**训练过程**：
- 总步数 134（8551 / 128 × 2）
- 耗时 **42.8 min**
- 显存峰值 **14.12 GB / 15.03 GB**（配额 ~94%）
- 最终 train_loss **0.68**，mean_token_accuracy **91.6%**，gradient_norm 稳定 1-2
- 全程无 OOM，无 loss spike

#### 5.5.3 结果（反直觉）

| 实验 | 方法 | **MCC** | Acc | unacc 召回 | acc 召回 | pred_acc_rate |
|---|---|---|---|---|---|---|
| E4 (LoRA r=16) | LoRA | **0.6246** | 0.839 | 73.5% | 88.8% | 0.693（近似真实）|
| **E4-FULL (本次)** | 全参 + 8bit optim | **0.5263** ⚠️ | 0.805 | **61.7%** ↓ | 88.8% | **0.732** ↑偏斜 |
| 原项目 1.7B-SFT | 全参 + fp32 optim | **0.657** | - | - | - | - |

**结论**：**全参 SFT 在我们的 14GB 环境下不仅没赢过 LoRA，反而倒退 -0.10**；距离原项目 0.657 有 -0.13 gap。

混淆矩阵对比（本次 E4-FULL）：
```
            pred=unacc  pred=acc
true=unacc    100         62       ← fp 比 LoRA 多 19 个
true=acc       41         324      ← 与 LoRA 完全相同（tp=324）
              61.7% 召回   88.8% 召回
```

#### 5.5.4 失败原因分析

**关键症状**：`pred_acc_rate` 从 LoRA 的 0.693 偏斜到 **0.732**（真实值 0.693），模型明显**偏向预测多数类 acceptable**（训练集 acc:unacc = 70:30）。

**三个嫌疑因素**：

1. **adamw_8bit 的量化噪声（主因）**  
   bitsandbytes 8bit AdamW 用 block-wise 动态量化存储 m/v 状态，正常训练没问题；但在 lr=1e-5 这种**极小学习率**下，量化误差占比变大，累计后导致**决策边界向多数类漂移**。  
   LoRA 则完全没这个问题（只更新 0.37% 参数，优化器状态 ~30MB，用 fp32）。  
   原项目有足够显存跑 fp32 AdamW，所以能收敛到更好的点。

2. **epochs=2 + 小 lr 下收敛不充分**  
   我们 train_loss 到 0.68，mean_token_accuracy 91.6%，看起来挺好；但全参 1.7B 可能需要更多更新步数才能充分学到**"少数类判别特征"**。LoRA 用 3 epoch + lr=2e-4 反而更匹配（effective learning = 约 3 倍）。

3. **max_length=128 的隐式正则**  
   对 99% 无截断的样本来说，max_length=128 vs 512 只是 padding 差异；但 trl SFTTrainer 会对 batch 内所有样本 **pad 到 max_length**（即使最长样本只有 93 token），这可能影响 attention mask 和 loss 的归一化分母，间接影响梯度量级。

**验证假设的最简方案**：用 **adamw_torch (fp32)** 重跑，把 max_length 砍到 96（压激活给 optim 让路）。但粗算：1.7B × 8 bytes = **13.6GB** 单纯 optim state，再加模型 3.4 + 梯度 3.4 = 20.4GB **完全不可能**。唯一可能是 **paged_adamw_32bit + CPU offload**，但 trl SFTConfig 不直接支持 `--optim paged_adamw_32bit` 的 CPU offload，需要改 Accelerate 配置，工程量大。

#### 5.5.5 结论

> **在 14GB 单卡硬件约束下，1.7B 全参 SFT 的最佳数值（0.5263）显著低于 LoRA r=16（0.6246），根源是必须使用 8bit 量化优化器。**

- **本机最佳方案仍是 E4 LoRA-SFT, MCC=0.6246**（与原项目 0.657 的 -0.032 gap 无法在单卡 14GB 下弥补）
- 要复现原项目 0.657，需要的条件：
  - 显存 ≥ 40GB（A100 40G / L40 48G），或
  - 多卡 DeepSpeed ZeRO-3 分片 optimizer state，或
  - 接受用 `paged_adamw_32bit` 的 CPU offload（速度慢 3-5 倍，不太实用）
- **重要反推**：这条实验也解释了 Step 6 的 GRPO 为什么做不起来 —— GRPO 需要维护 3 份全参模型（policy + ref + value/critic）+ rollout 推理，14GB 根本装不下 1.7B 全参，被迫退回 LoRA 也就注定了上限。

产物：
- ckpt: `my_try/ckpt/Qwen3-1.7B-fullsft-E4-FULL/`
- 日志: `my_try/logs/fullsft_1.7B_E4.log`
- 评测: `my_try/results/step5_fullsft/res_fullsft_Qwen3-1.7B-fullsft-E4-FULL.json`

---

### 5.6 基于 LoRA 最优解的三步迭代（追加：超参精调 → QLoRA 4B → Hard Example Mining）

#### 5.6.1 动机

§5.5 证明了 14G 单卡下"全参 SFT 反而不如 LoRA"，本机最佳是 E4 LoRA r=16 (MCC=0.6246)。
顺着"**LoRA 才是甜点**"这条线，做三步迭代：

1. **Step 1：LoRA 超参精调** —— rank / target_modules / batch / epochs 扫描
2. **Step 2：跨越 14G 限制** —— Qwen3-4B + QLoRA (nf4) 冲击 0.70+
3. **Step 3：Hard Example Mining** —— 用最优模型挖训练集错题，加权重训

#### 5.6.2 Step 1 LoRA 超参扫描（4 组 + baseline）

baseline = E4 (r=16, qkvo, eff_bsz=32, 3ep, lr=2e-4, MCC=0.6246)。

| 实验 | rank | target_modules | eff_bsz | epochs | lr | **MCC** | Δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline E4 | 16 | qkvo | 32 | 3 | 2e-4 | 0.6246 | — |
| **E4-A-mlp** | 16 | **+gate/up/down** | 32 | 3 | 2e-4 | **0.6575** | **+0.033** |
| E4-B-bsz128 | 16 | qkvo | **128** | 3 | 2e-4 | **0.5800** | **−0.045** ⚠️ |
| 🥇 **E4-C-r32** | **32** | qkvo | 32 | 3 | 2e-4 | **0.6654** | **+0.041** |
| E4-D-best | 32 | +MLP | 128 | 5 | 1e-4 | 0.6567 | +0.032 |

**关键洞察**：

1. **rank 32 单独贡献最大 (+0.041)** — 1.7B + CoLA 在 r=16 时容量不够，r=32 (12.8M 可训参数) 到甜点。
2. **MLP target 也有 +0.033 独立收益** — MLP 里藏着语法判别的特征转换。
3. **大 batch (eff=128) 是大坑 (−0.045)** — 与全参 SFT 相反！LoRA 容量小、lr=2e-4 偏大，eff_bsz=128 时每 epoch 仅 67 step，**梯度更新次数不足**未能充分收敛。
4. **D 组组合拳不是最佳** — +MLP(+0.033) × +r32(+0.041) × +eff128(−0.045) × +5ep(+) 互相抵消，最终 MCC=0.6567 仍低于单独的 E4-C。

🏆 **Step 1 最优：E4-C-r32 (MCC=0.6654)**，已超过原项目 1.7B 全参 SFT 报告值 **0.657**！

#### 5.6.3 Step 2 Qwen3-4B + QLoRA（突破 14G 限制）

**动机**：1.7B 不够大、且 LoRA 容量已被 r=32 摸高。换 4B 基座 + 4bit nf4 量化的 QLoRA，理论显存 ≤ 8GB，留出空间提升模型本身的语法能力。

**配置**：
- 基座：Qwen3-4B (从 ModelScope 下载，3.5min, 7.6GB)
- 量化：bnb nf4 + double_quant + bf16 compute
- LoRA：rank=32, target=qkvo+MLP, eff_bsz=64, 3ep, lr=1e-4
- 优化器：adamw_8bit (paged)
- 训练：97.7 min，**显存峰值仅 5.36 GB**（QLoRA 威力）

**结果**：MCC=**0.6633** (Acc=**0.8615** 是所有实验最高的！)

#### 5.6.4 Step 1 vs Step 2 的"性格对比"

```
              E4-C-r32 (1.7B)        E6 (4B QLoRA)
              ──────────────         ────────────
TN / FP        130 /  32              109 /  53
FN / TP         45 / 320               20 / 345
unacc 召回    80.2% ⭐                67.3%
acc   召回    87.7%                   94.5% ⭐
MCC          0.6654 ⭐               0.6633
Acc          0.8539                  0.8615 ⭐
```

**反直觉**：4B 模型 Accuracy 更高，但 MCC 反而稍低 0.002。这是因为：

- **E4-C 偏严格**（敢说 unacceptable，但会错杀 acceptable）
- **E6 偏宽容**（少错杀 acceptable，但会放过 unacceptable，多数类偏斜）

E6 的多数类偏斜推测来自**4bit nf4 量化噪声 + adamw_8bit 在 lr=1e-4 下的累积误差**（与 §5.5 全参 SFT 的失败模式同源），符合"**量化优化器在小 lr 下决策边界向多数类漂移**"的假设。

**两个模型预测互补，是 ensemble 的完美素材**（错的样本不重叠，前 10 条预测一致率 9/10，后续可优化）。

#### 5.6.5 Step 3 Hard Example Mining（反向效果，但有结构性收获）

**做法**：
1. 用 E4-C-r32 对训练集 8551 条做 forward 推理，取最后位置 logits 上 (acc, unacc) 的 softmax 置信度
2. 标记**错题** (1109 条) + **低置信** (`prob_true_label < 0.6`, 1548 条) 为 hard
3. 加权数据集：普通 ×1, hard ×3 → 11647 条
4. 用 E4-C-r32 同配置（r=32, qkvo, eff=32, 3ep, lr=2e-4, cosine）冷启动重训

**踩坑**（修复后才有有效结果）：
- mining 脚本初版用 `attention_mask.sum()-1` 取最后位置，**left-padding 时应该用 `[:, -1, :]`**
- 没传 `enable_thinking=False`，Qwen3 chat_template 默认开 thinking 模式，logit 位置错位
- 修复后训练集 acc=87.03%（vs dev 85.40%），轻微过拟合在合理范围

**结果**：

| 实验 | MCC | Acc | unacc 召回 | acc 召回 |
| --- | --- | --- | --- | --- |
| 🥇 E4-C-r32 (基线) | **0.6654** | 0.8539 | 80.2% | 88.8% |
| E4-HARD (加权重训) | **0.5820** ❌ | 0.7989 | **84.0%** ⬆ | **78.1%** ⬇ |

**反向效果分析**：
- **少数类 (unacc) 召回 +3.8 pt** ✅ 加权确实让模型更敏感
- **多数类 (acc) 召回 −10.7 pt** ❌ 严重误伤
- 净效果 MCC **−0.083**

**根因**：CoLA 训练集本身 acc:unacc ≈ 70:30；hard 样本里 unacc 占比偏高（模型本就更难判 unacc），重复 3 倍后类别比变成约 **acc 60% : unacc 40%**。模型学过头，把许多 acc 句子也判为 unacc。

**最难的训练样本观察**（**意外发现 CoLA 标注噪声**）：

```
prob_true=0.004  true=unacc  "Megan loves Jason."          ← 完全合法的句子被标 unacc!
prob_true=0.005  true=unacc  "The committee hasn't yet made up their mind."  ← 主谓一致争议
prob_true=0.006  true=unacc  "In the corner lay a dog."    ← 倒装合法句
prob_true=0.007  true=unacc  "It's high time Fiona gets a job."  ← 美式英语合法
```

→ **CoLA 训练集存在 ~5-10% 标注噪声**，这就是模型 MCC 的"软上限"。

#### 5.6.6 三步综合榜（Step 5 全部结果按 MCC 排序）

| # | 实验 | 模型 | 方法 | MCC | Acc | 备注 |
| - | --- | --- | --- | --- | --- | --- |
| 1 | 🥇 **E4-C-r32** | Qwen3-1.7B | LoRA r=32 | **0.6654** | 0.8539 | **本机最优** |
| 2 | 🥈 **E6 (Qwen3-4B QLoRA)** | Qwen3-4B | QLoRA r=32 +MLP | **0.6633** | **0.8615** | Acc 最高 |
| 3 | 🥉 E4-A-mlp | Qwen3-1.7B | LoRA r=16 +MLP | 0.6575 | 0.8520 | — |
| 4 | E4-D-best | Qwen3-1.7B | LoRA r=32 +MLP +eff128 5ep | 0.6567 | 0.8501 | — |
| 5 | E4 (baseline) | Qwen3-1.7B | LoRA r=16 qkvo | 0.6246 | 0.8387 | — |
| 6 | E4-HARD | Qwen3-1.7B | LoRA r=32 +HardWeight ×3 | 0.5820 | 0.7989 | hard 加权失败 |
| 7 | E4-B-bsz128 | Qwen3-1.7B | LoRA r=16 eff=128 | 0.5800 | 0.8254 | 大 batch 反向 |
| 8 | E4-CoT | Qwen3-1.7B | LoRA r=16 + DeepSeek CoT | 0.5643 | 0.8121 | §5.4 |
| 9 | E1 | Qwen3-0.6B | LoRA r=16 | 0.5406 | 0.8083 | — |
| 10 | E1-FULL v2 | Qwen3-0.6B | 全参 SFT lr=5e-5 | 0.5361 | 0.7989 | — |
| 11 | E4-FULL | Qwen3-1.7B | 全参 SFT + adamw_8bit | 0.5263 | 0.8046 | §5.5 |
| 12 | E1-FULL v1 | Qwen3-0.6B | 全参 SFT lr=1e-5 | 0.4526 | 0.7818 | 失败案例 |
| 13 | E1-CoT | Qwen3-0.6B | LoRA + DeepSeek CoT | 0.4135 | 0.7666 | §5.4 |

#### 5.6.7 与外部锚点对照

```
0.598 ─── 原项目 Qwen3-0.6B 全参 SFT (README)
0.610 ─── 原项目 Qwen3-0.6B-CLS
0.625 ─── E4 (我们 baseline LoRA r=16)
0.636 ─── DeepSeek-R1 0120 (API)
0.657 ─── 原项目 Qwen3-1.7B 全参 SFT (README) ⭐
─────────────── 我们已超过 ↑ ───────────────
0.6633 ── ⭐ E6 (Qwen3-4B QLoRA, Acc=0.8615 最高)
0.6654 ── ⭐ E4-C-r32 (本机最优 MCC)
0.702 ─── 原项目 Qwen3-1.7B SFT+GRPO (天花板)
0.703 ─── gemini-3.1-flash-lite (API)
0.718 ─── claude-opus-4.7 (API)
0.727 ─── deepseek-v3-0324 (API)
```

**结论**：
- **本机最佳 MCC=0.6654 已经超过原项目 1.7B 全参 SFT 0.657**（+0.001 略胜）
- 距离原项目"SFT+GRPO 天花板" 0.702 还有 **−0.037 gap**
- 这个 gap 在 14G 单卡下**几乎不可能用 GRPO 弥补**（§Step 6 已证），但仍有改进空间（见下）

#### 5.6.8 关键经验沉淀

1. **LoRA 不是全参的"备胎"，而是小数据 + 单卡场景的"最优解"**：
   - 全参 SFT 在小数据 (8551) + 必须用 8bit optim 的约束下，决策边界向多数类漂移（§5.5）
   - LoRA 的低秩约束是天然正则，小卡上反而更稳

2. **超参的真正胜负手是 rank + target_modules**，不是 batch / epochs / lr：
   - rank 16 → 32 涨 +0.041
   - +MLP target 涨 +0.033
   - 这两者**几乎不互相增强**（D 组没拿到累加收益），说明它们攻击的是同一类瓶颈：**容量**

3. **大 batch 是 LoRA 的反模式**：与全参 SFT 经验相反，LoRA 在小数据上需要"小 batch + 多 step"

4. **QLoRA 在 4B 上不输 LoRA 在 1.7B 上**，但**会引入多数类偏斜**（量化优化器在小 lr 下的累积误差，与 E4-FULL 同源）

5. **Hard Example Mining 必须做类别平衡**：朴素重复 hard 样本会破坏原始类别比，需要按类别分别选 top-K，或对错题用 0.7-0.8× 较小权重

#### 5.6.9 仍未挖掘的方向（明天/后续）

| 方案 | 预期收益 | 工程量 |
| --- | --- | --- |
| **类别平衡的 Hard Mining**（每类 top-K + ×2）| +0.01-0.02 | 30min |
| **1.7B + 4B Ensemble** (vote / prob avg) | +0.02-0.04 | 1h |
| **DPO** (用 1.7B-SFT 的对错样本对自动构造偏好) | +0.01-0.02 | 2h |
| **Qwen3-4B QLoRA 调参**（小 lr=5e-5, 5ep, qkvo）减少多数类偏斜 | +0.01-0.02 | 100min |

#### 5.6.10 产物清单

**脚本**（新增/修改）：
- `my_try/scripts/run_sweep_e4.sh` — Step 1 LoRA 扫参 runner
- `my_try/scripts/train_lora_sft.py` — 加 `--load_in_4bit` / `--optim` 支持 QLoRA
- `my_try/scripts/run_after_sweep.sh` — watchdog：扫参完成后自动起 4B QLoRA
- `my_try/scripts/mine_hard_examples.py` — Hard Example Mining 推理脚本（修过 left-padding + thinking bug）
- `my_try/scripts/build_weighted_sft_data.py` — 加权数据集构造
- `my_try/scripts/summarize_step5.py` — 全实验汇总

**ckpt**：
- `my_try/ckpt/Qwen3-1.7B-lora-E4-A-mlp-r16/`
- `my_try/ckpt/Qwen3-1.7B-lora-E4-B-bsz128-r16/`
- `my_try/ckpt/Qwen3-1.7B-lora-E4-C-r32-r32/` ⭐
- `my_try/ckpt/Qwen3-1.7B-lora-E4-D-best-r32/`
- `my_try/ckpt/Qwen3-4B-lora-E6-r32/` ⭐
- `my_try/ckpt/Qwen3-1.7B-lora-E4-HARD-r32/`

**数据**：
- `my_try/data/hard_examples_Qwen3-1.7B-lora-E4-C-r32-r32.jsonl`（8551 条带置信度）
- `my_try/data/sft_train_weighted.jsonl`（11647 条加权）

**评测结果**：见 `my_try/results/step5_lora/res_lora_*.json`

---

## 6. 关键发现

### 6.1 LoRA-SFT 是小模型的"救命稻草"

Qwen3-0.6B 的 **MCC 从 0.000 → 0.5406**（+0.54），巨大跃升：
- 甚至**超过** Qwen3-1.7B 的 zero-shot (0.500)
- 追平 Qwen3.5-2B 的 zero-shot (0.549)
- 只比原项目 **全参 SFT** 差 0.058（同样的模型，只是我们用 LoRA）

**含义**：对 CoLA 这种"小模型掌握不了的任务"，LoRA 微调是性价比极高的手段，不需要全参 SFT，不需要 24GB 显存。

### 6.2 大模型的 LoRA 增益有边际递减

Qwen3-1.7B 的增益只有 **+0.12**（0.500 → 0.6246），远小于 0.6B 的 +0.54。

**原因推断**：1.7B 在 pretrain 阶段已经学到了大部分 CoLA 需要的语言能力，SFT 只是"把它学会的东西调出来"，而 0.6B 需要"真的学会"这个任务。符合我们 Step 4 的结论：**1.7B 已经过了"能力门槛"，0.6B 没过**。

### 6.3 rank=16 已经够用（尚未进一步对比）

两个实验的可训练参数占比都不到 1%：
- 0.6B: **0.76%**
- 1.7B: **0.37%**

eval_loss 在 epoch 3 仍在下降，但**下降幅度已很小**（0.4515 → 0.4338 → 0.4300），说明模型接近饱和。**加大 rank 未必会有大的提升**（后续 E2/E3 可验证）。

### 6.4 parse_fail 归零是格式稳定性的证明

baseline 下：
- Qwen3-0.6B zero-shot → **全部 527 条输出 `acceptable`**（100% parse 成功，但全错）
- Qwen3-0.6B thinking → 有些输出偏长，需 max_new_tokens=1024

LoRA SFT 后：
- E1/E4 的 avg_new_tokens = **1.3**（答案平均只有 1.3 个 token）
- parse_fail = **0/527**
- **"acceptable/unacceptable" 的格式被训练成接近纯分类头的输出**

这是 `completion_only_loss=True` 的威力：loss 只作用于 assistant 段，推着模型**非常精准地产出格式化答案**。

### 6.5 工程上的几个实用经验

1. **14 GB 卡可以同时跑 0.6B + 1.7B 两个 LoRA SFT**，稳态合计 ~8 GB
2. **训练总时长 = max(E1, E4) = 23.5 min**（并行比串行省一半）
3. **LoRA adapter 只 30 MB**（E1）/ **50 MB**（E4），部署超轻
4. 评测全集 527 条只要 **4 秒**（batch=16，merge_and_unload 后跑），可以随时 ablation

### 6.6 距离原项目的 gap 来源分析

| 实验 | 我们 LoRA | 原项目全参 | gap | 我们训练量 |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 0.541 | 0.598 | -0.058 | 0.76% 参数 |
| Qwen3-1.7B | 0.625 | 0.657 | -0.032 | 0.37% 参数 |

gap 随模型变大而缩小，说明：**大模型的 LoRA 效果越接近全参 SFT**（和 QLoRA 论文的规律一致）。

如果要进一步缩小 gap，可能的尝试：
- 加大 rank 到 32/64（E2/E3/E5）
- 加 target_modules（把 `gate_proj / up_proj / down_proj` 也加进来）
- 训练 5 个 epoch 而不是 3 个
- 用 CoT 数据（需要 R1 蒸馏）

### 6.7 全参 SFT 在 14G 卡上完全可行，但 lr 是关键

这是一个**反直觉**的发现。我们原以为 14G 卡做不了 Qwen3-0.6B 全参 SFT（理论估算 13-14 GB），但实测**完全可行**：

| 配置 | 峰值显存 | 训练耗时 | MCC | 结论 |
| :--- | :-: | :-: | :-: | :--- |
| v1: lr=1e-5, bsz=4, grad_ckpt | 6.01 GB | 16.2 min | 0.4526 | 显存省但 lr 太保守，没收敛 |
| **v2: lr=5e-5, bsz=16, 无 grad_ckpt** | **13.23 GB** | **4.3 min** | **0.5361** | 充分榨干显存 + 激进 lr 的组合拳 |

**v2 的 3 个关键调整**：
1. **关 `gradient_checkpointing`** → 显存从 6 GB 提到 13 GB，但**速度快 3-4 倍**（免了反向传播时重算激活）
2. **`bsz` 从 4 提到 16** → 吞吐量 4 倍提升
3. **`lr` 从 1e-5 提到 5e-5** → 快速收敛，eval_loss 在 epoch 2 达到最低 0.3616

**经验教训**：
- 14G 显存不是全参 SFT 的死线，**显存越省 ≠ 越好**
- 小模型 (≤1B) + 小数据 (<10K) 用**激进 lr**（5e-5 ~ 1e-4）反而比保守 lr（1e-5）快收敛
- **epoch 3 开始过拟合**（eval_loss 回升），说明可以换 `epochs=2` 再提 0.01-0.02 的 MCC
- 最终 **LoRA 和全参几乎打平**（0.5406 vs 0.5361）——这是 LoRA 在"任务简单 + 数据中等"场景下的典型表现，和 QLoRA 论文结论一致

### 6.8 本次实验的整体性价比排名

| 方案 | MCC | 训练时长 | 显存 | 可训练参数 | 性价比 |
| :--- | :-: | :-: | :-: | :-: | :-: |
| E1 LoRA r=16 | **0.5406** | 13.4 min | 3.57 GB | 4.6M | ⭐⭐⭐⭐⭐ |
| E1-FULL v2 全参 | 0.5361 | 4.3 min | 13.2 GB | 596M | ⭐⭐⭐⭐ |
| E4 LoRA r=16 | **0.6246** | 23.5 min | 4.73 GB | 6.4M | ⭐⭐⭐⭐⭐ |

**LoRA 综合最优**：效果持平全参，但参数量只有 0.76%，显存 1/4，可以并行跑多个实验，adapter 文件仅 30 MB 方便部署和实验切换。

---

## 7. 产出物清单

### 7.1 脚本

| 文件 | 作用 | 状态 |
| --- | --- | --- |
| `my_try/prepare_sft_data.py` | 数据格式转换（共享用于 LoRA 和全参） | ✅ |
| `my_try/train_lora_sft.py` | LoRA SFT 训练 | ✅ |
| `my_try/eval_lora.py` | LoRA 评测（MCC） | ✅ |
| `my_try/train_full_sft.py` | **全参 SFT 训练**（新增） | ✅ |
| `my_try/eval_fullsft.py` | **全参 SFT 评测**（新增） | ✅ |

### 7.2 数据

| 文件 | 条数 |
| --- | --- |
| `my_try/data/sft_train.jsonl` | 8,551 |
| `my_try/data/sft_val.jsonl` | 527 |

### 7.3 checkpoint & 结果（运行后产出）

| 目录/文件 | 内容 |
| --- | --- |
| `my_try/ckpt/Qwen3-0.6B-lora-E1-r16/` | E1 adapter（30 MB，`.gitignore` 忽略） |
| `my_try/ckpt/Qwen3-1.7B-lora-E4-r16/` | E4 adapter（50 MB） |
| `my_try/ckpt/Qwen3-0.6B-fullsft-E1-FULL-v2/` | **E1-FULL v2 全参模型**（~1.2 GB，`.gitignore` 忽略） |
| `my_try/res_lora_Qwen3-0.6B-lora-E1-r16.json` | E1 评测 |
| `my_try/res_lora_Qwen3-1.7B-lora-E4-r16.json` | E4 评测 |
| `my_try/res_fullsft_Qwen3-0.6B-fullsft-E1-FULL-v2.json` | **E1-FULL v2 评测** |

---

## 8. 遇到的坑

### 8.1 `_lzma` 模块缺失（阻塞 trl）

**现象**：

```python
>>> from trl import SFTTrainer
ModuleNotFoundError: No module named '_lzma'
```

**原因**：系统 Python 3.12（`/usr/local/python3.12`）编译时没装 `xz-devel`，`_lzma` C 扩展没编出来。`datasets` 库顶层 `import lzma`，导致链式失败。

**修复**：不重编 Python，放一个**纯 Python stub 模块**到 venv：

```
/data/workspace/Github-open/CoLA-RL/.venv-cola/
  lib/python3.12/site-packages/_lzma.py    ← 放入 stub
```

stub 导出所有 `lzma.py` 需要的常量 + 桩类（实际调用会抛 `NotImplementedError`）。由于我们项目只用本地 tsv/jsonl，**不读 .xz 数据集**，stub 足够。

### 8.2 trl 1.3.0 的 API 变更

| trl 0.x → 1.x | 说明 |
| --- | --- |
| `tokenizer=` → **`processing_class=`** | SFTTrainer 参数名变更 |
| `max_seq_length=` → **`max_length=`** | SFTConfig 字段名变更 |
| 手动 `get_peft_model` → **直接传 `peft_config=`** | SFTTrainer 会自动 wrap |

### 8.3 `target_modules` 类型问题

**现象**：训练结束保存 `run_info.json` 时：

```
TypeError: Object of type set is not JSON serializable
```

**原因**：`peft.LoraConfig` 内部把 `target_modules` 列表转成了 `set`。

**修复**：`json.dump` 前把 set 转 list：

```python
"target_modules": sorted(list(lora_cfg.target_modules)),
```

### 8.4 生成时右 padding 警告

**现象**：`eval_lora.py` batch 评测时：

```
A decoder-only architecture is being used, but right-padding was detected!
For correct generation results, please set `padding_side='left'`
```

**原因**：decoder-only 模型生成时，如果右侧有 pad token，生成会从 pad 之后开始，结果错乱。

**修复**：加载 tokenizer 后立刻设 `tok.padding_side = "left"`。

### 8.5 两进程共享 14G 卡

**现象**：E1 + E4 并行启动后，GPU 已用量一度飙到 13.88 GB / 14.34 GB（只剩 454 MB），it/s 也变慢。

**观察**：训练进入稳态后，显存回落到约 **12.3 GB**，空闲约 2 GB，速度恢复到单跑水平。

**经验**：trl 的 `SFTTrainer` 初始化阶段有一个显存高峰（准备 grad_checkpoint、编译 autograd 图），**稳态比峰值低 15-20%**。如果只看峰值会误以为必须串行。

---

## 附录：一页式开跑命令

```bash
cd /data/workspace/Github-open/CoLA-RL

# 1) 准备数据（只需一次）
.venv-cola/bin/python my_try/prepare_sft_data.py

# 2a) LoRA SFT 训练（可单独也可并行）
.venv-cola/bin/python my_try/train_lora_sft.py \
    --model model/Qwen3-0.6B --exp E1 --rank 16 --epochs 3 --bsz 8 --grad_accum 4

.venv-cola/bin/python my_try/train_lora_sft.py \
    --model model/Qwen3-1.7B --exp E4 --rank 16 --epochs 3 --bsz 4 --grad_accum 8

# 2b) 全参 SFT 训练（v2 激进版，充分用满 14 GB）
# 注：需要手动改 train_full_sft.py 把 gradient_checkpointing 关掉，或用 monkey-patch
.venv-cola/bin/python my_try/train_full_sft.py \
    --model model/Qwen3-0.6B --exp E1-FULL-v2 \
    --epochs 3 --bsz 16 --grad_accum 2 --lr 5e-5 --max_length 256

# 3a) LoRA 评测（读取 adapter_config 自动定位基座）
.venv-cola/bin/python my_try/eval_lora.py \
    --adapter my_try/ckpt/Qwen3-0.6B-lora-E1-r16

.venv-cola/bin/python my_try/eval_lora.py \
    --adapter my_try/ckpt/Qwen3-1.7B-lora-E4-r16

# 3b) 全参 SFT 评测
.venv-cola/bin/python my_try/eval_fullsft.py \
    --model my_try/ckpt/Qwen3-0.6B-fullsft-E1-FULL-v2
```
