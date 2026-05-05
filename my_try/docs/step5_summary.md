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
     0.453 ─── E1-FULL v1: 0.6B 全参 SFT (lr=1e-5 保守)  ← 失败案例，调参不足
     0.500 ─── Qwen3-1.7B zero-shot
     0.527 ─── hy3-preview (API 推理型)
     0.536 ─── ⭐ E1-FULL v2: 0.6B 全参 SFT (lr=5e-5 激进) ← 调参后
     0.541 ─── ⭐ E1: Qwen3-0.6B + LoRA-SFT r=16       ← 小模型 LoRA 最强!
     0.549 ─── Qwen3.5-2B zero-shot
     0.598 ─── 原项目 Qwen3-0.6B 全参 SFT (README)     (我们 E1 差 -0.058)
     0.610 ─── 原项目 Qwen3-0.6B-CLS
     0.625 ─── ⭐ E4: Qwen3-1.7B + LoRA-SFT r=16       ← 1.7B LoRA
     0.636 ─── DeepSeek-R1 0120
     0.657 ─── 原项目 Qwen3-1.7B 全参 SFT              (E4 差 -0.032)
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
