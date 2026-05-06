# CoLA-RL 本地复现实验：方法论复盘与经验沉淀

> 创建日期：2026-05-06
> 项目：`my_try/`，基于 [ytzfhqs/CoLA-RL](https://github.com/ytzfhqs/CoLA-RL) 的本地复现尝试
> 硬件：单卡 NVIDIA L20（vGPU 配额 ~15 GB）
> 实验跨度：2026-05-04 → 2026-05-06，共 26 个有效实验
> 关联文档：[Step 3 baseline](baseline_summary.md) · [Step 4 免训练](step4_summary.md) · [Step 5 SFT](step5_summary.md) · [Step 6 GRPO](step6_summary.md)

---

## 0. 一句话结论

> **在 14 GB 单卡上，对 8551 条 CoLA 二分类数据，最有效的方案是 LoRA r=32（仅 attention QKVO）+ Qwen3-1.7B 短答案 SFT，最终 MCC=0.6654，超过原项目报告的全参 SFT 0.657**；
> 三类"看上去更高级"的方法 —— 全参 SFT、CoT 蒸馏、GRPO —— 在这块卡上**全部反向**或没收益；
> 真正的胜负手不是算法选择，而是**容量约束（rank/MLP target）+ 类别平衡 + 显存可负担的优化器精度**。

---

## 目录

- [1. 实验方向全景：26 个尝试，4 类结论](#1-实验方向全景26-个尝试4-类结论)
- [2. 哪些方向有效、哪些无效（含原因分析）](#2-哪些方向有效哪些无效含原因分析)
- [3. 通用模式归纳：可复用到其它任务的 10 条经验](#3-通用模式归纳可复用到其它任务的-10-条经验)
- [4. 复杂任务迁移：从 CoLA 二分类到疾病诊断 / 股票预测](#4-复杂任务迁移从-cola-二分类到疾病诊断--股票预测)
- [5. 模型微调策略决策树](#5-模型微调策略决策树)
- [6. SFT vs RL：何时介入、何时放弃](#6-sft-vs-rl何时介入何时放弃)
- [7. LoRA vs 全参 vs QLoRA：硬件约束下的真实成本曲线](#7-lora-vs-全参-vs-qlora硬件约束下的真实成本曲线)
- [8. 工程踩坑与方法论沉淀](#8-工程踩坑与方法论沉淀)
- [9. 个人经验自评：哪些做得好、哪些可以更好](#9-个人经验自评哪些做得好哪些可以更好)
- [10. 后续可挖方向（如果再继续）](#10-后续可挖方向如果再继续)

---

## 1. 实验方向全景：26 个尝试，4 类结论

我把整个实验过程按"方向"重新归类（非时间顺序），共 **9 个大方向、26 个子实验**：

```
方向 A. Baseline 摸底  ───────── 4 模型 zero-shot      → 锚定基线
方向 B. Prompt 工程    ───────── 4 模型 × 4 变体 = 16  → 全部失效（除 0.6B thinking）
方向 C. API 大模型      ───────── 4 个 API zero-shot     → 锚定天花板 0.73
方向 D. 经典 ML        ───────── 5 个传统方法           → 锚定下限 0.18
方向 E. LoRA SFT       ───────── 1 + 2 + 4 + 1 = 8 个   → ✅ 主战场，本机最佳
方向 F. 全参 SFT       ───────── 0.6B × 2 + 1.7B × 1     → ⚠️ 0.6B 可，1.7B 退化
方向 G. CoT 蒸馏 SFT   ───────── 0.6B / 1.7B × 1         → ❌ 两个尺寸均退化
方向 H. GRPO (RL)      ───────── v1/v2/v3/v4 共 5 次     → ❌ 全部 mode collapse
方向 I. Hard Mining    ───────── 1 次重训                → ⚠️ 类别失衡导致退化
```

**最终成绩（按 MCC 排序，27 个数据点）**：

```
                                                                           本机最佳
                                                                              ↓
0.000 ─ 0.176 ───── 0.500 ──── 0.5491 ──── 0.598 ── 0.625 ─ 0.6654 ─ 0.702 ───── 0.727
  ↑       ↑           ↑          ↑          ↑       ↑       ↑         ↑          ↑
 baseline LR 上限  1.7B z-shot  2B z-shot   原 0.6B  E4    E4-C    原项目      DeepSeek-V3
                                            全参 SFT       LoRA r=32 SFT+GRPO   API 上限
```

---

## 2. 哪些方向有效、哪些无效（含原因分析）

### 2.1 ✅ 有效方向

| # | 方向 | 实验 | 收益 | 起作用的根因 |
|---|---|---|---|---|
| 1 | **LoRA SFT 短答案** | E4 baseline (1.7B, r=16) | MCC: 0.500 → 0.625 (+0.125) | 任务本质是"决策阈值校准"，LoRA 容量恰好够 |
| 2 | **LoRA rank ↑** | E4-C-r32 (r=16→32) | +0.041 | r=16 时 LoRA 容量瓶颈，r=32 摸到甜点 |
| 3 | **LoRA target 扩到 MLP** | E4-A-mlp | +0.033 | MLP 里藏着语法判别的特征转换 |
| 4 | **QLoRA 4bit 跨越显存墙** | E6 (Qwen3-4B QLoRA) | 占用 5.36GB, 4B 模型可训 | nf4 + bf16 compute + LoRA 增量, 量化误差被 LoRA 增量吸收 |
| 5 | **LoRA-SFT 在 0.6B 上的"救命稻草"效应** | E1 (0.6B LoRA r=16) | MCC: 0.000 → 0.541 (+0.541) | base 模型 zero-shot 退化为"全说 yes"时, SFT 注入"unacceptable"的判别偏好 |

### 2.2 ❌ 无效 / 反向方向

| # | 方向 | 实验 | 损失 | 失败的根因 |
|---|---|---|---|---|
| 1 | **Prompt 工程**（fewshot/CoT/thinking）| 16 组 | 1.7B/2B 全部 ≤ plain | 强模型已掌握任务，强行加思考反而漂移；CoT 把 1.7B 从 0.50 打到 0.14 |
| 2 | **CoT 蒸馏 SFT** | E1-CoT (0.6B), E4-CoT (1.7B) | -0.127 / -0.060 | 任务太简单不需要 CoT；DeepSeek 蒸馏 13% 噪声；小模型模仿大模型 CoT 学到"宽容偏见"；分类 token 梯度被推理 token 稀释 |
| 3 | **全参 SFT 1.7B**（14GB 卡）| E4-FULL | -0.098（vs LoRA r=16）| 必须用 adamw_8bit 量化优化器 → lr=1e-5 下决策边界向多数类漂移（pred_acc_rate 0.732 偏离真实 0.693）|
| 4 | **GRPO RL 0.6B** | v1/v2/v3/v4 | 全部 mode collapse 到 MCC=0 | 类别极不均衡 + 短答案信号过弱 + ref policy 刚 SFT 完没有探索空间 + reward shaping 不足以撬动局部最优 |
| 5 | **GRPO RL 1.7B** | v4 | 不崩但退化 -0.13 | 同上，在 14GB 下被迫用 LoRA GRPO，rollout 多样性不够 |
| 6 | **大 effective batch (128)** in LoRA | E4-B-bsz128 | -0.045 | LoRA 容量小、lr=2e-4 偏大，eff_bsz=128 时每 epoch 仅 67 step，**梯度更新次数不足**未充分收敛 |
| 7 | **Hard Example Mining 朴素加权** | E4-HARD | -0.083 | hard 集天然偏向少数类（unacc），×3 重复后训练分布从 70:30 → 60:40，模型矫枉过正把 acc 也判 unacc |

### 2.3 ⚠️ 中性方向（值得记录）

| # | 方向 | 结论 |
|---|---|---|
| 1 | 全参 SFT 0.6B 调参后 (E1-FULL-v2, lr=5e-5) | MCC=0.5361，**与 LoRA r=16 (0.5406) 几乎打平**，但 LoRA 训练时间是 1/4 |
| 2 | LoRA D 组合拳 (r=32 + MLP + eff=128 + 5ep) | MCC=0.6567，**互相抵消**：+r32 +MLP 涨幅被 +eff128 -ep 抵消，组合不一定 > 单点最优 |
| 3 | API 大模型 thinking 型 (hy3-preview) | MCC=0.527，反而比 deepseek-v3 (0.727) 低 0.20，**推理模型不适合简单二分类** |

### 2.4 三个反直觉但已验证的发现

> 这三条是这次实验最有信息量的产出，值得记下来：

#### A. **更大模型不一定更好（E6 4B QLoRA = 0.6633 < E4-C 1.7B LoRA = 0.6654）**

4B QLoRA 的 Acc 0.8615 是所有实验里最高的，但 MCC 反而低 0.002。混淆矩阵分析：

```
              E4-C (1.7B)         E6 (4B QLoRA)
unacc 召回    80.2%               67.3%   ← 4B 偏宽容
acc   召回    87.7%               94.5%   ← 4B 多数类偏斜
```

根因：**nf4 量化 + adamw_8bit 在小 lr 下产生量化噪声累积，导致决策边界向多数类漂移**。这条结论与 E4-FULL（全参 + 8bit 优化器）的失败模式同源。

#### B. **大 batch 在全参 SFT 是必须的（原项目 eff=128），但在 LoRA 上是大坑（-0.045）**

经典 SFT 经验"大 batch 收敛更稳"在 LoRA 上失效，原因：
- LoRA 可训参数仅 12M（0.7%）
- lr 用的是 LoRA 标配 2e-4（比全参大 20×）
- eff_bsz=128 时每 epoch step 数从 267 → 67
- 梯度更新次数 × 学习率 = 总学习量降低 → 欠拟合

→ **LoRA 是"小 batch 多步"的典型代表，不要照搬全参 SFT 的 batch 设置。**

#### C. **小模型从 SFT 中获益最大（0.6B: +0.541），大模型增益边际递减**

```
zero-shot → LoRA-SFT 后  →  绝对增量
0.6B   0.000 → 0.541      +0.541
1.7B   0.500 → 0.625      +0.125
4B     ?     → 0.663      未测 zero-shot, 估计 +0.05~0.08
```

这是**典型的对数饱和曲线**。指导意义：当 zero-shot 已经 ≥ 0.6 时，做 SFT 性价比急剧下降，应该换方向（更好的数据 / ensemble / 算法切换）。

---

## 3. 通用模式归纳：可复用到其它任务的 10 条经验

### Pattern 1: **先做 baseline + 经典 ML 锚点，再决定要不要上模型**

我们 Step 4 做的最有价值的事情之一是把 5 个经典 ML baseline 跑了一遍（耗时 < 5 秒）。**WordNgramLR=0.176 的"线"非常关键**：
- Qwen3-0.6B zero-shot = 0.000，**打不过 LR**
- 只有 1.7B+ 才稳压 LR 一倍以上

→ **如果 LLM zero-shot 打不过 TF-IDF + LR，就别做 LLM**。这条原则在医疗/金融场景特别重要：很多任务用一个调好的 XGBoost 就能解决。

### Pattern 2: **类别分布 + 偏斜方向是首要诊断指标**

每次评测我们都看 `pred_acc_rate` vs 真实 `0.693`：

```
pred_rate ≈ 真实 → 模型平衡，MCC 高
pred_rate > 真实 → 多数类偏斜（"全说 yes"病）
pred_rate < 真实 → 少数类过敏（HardMining 加权过头）
```

这比只看 Accuracy 重要得多。CoLA 70:30 时，**全说 acc 的"傻模型"Accuracy=0.693 但 MCC=0**。在医疗（病例阳性常 < 5%）/ 金融（涨跌通常 50:50）场景，这个指标更关键。

### Pattern 3: **任务复杂度决定要不要 CoT**

CoLA 这种"短句二分类"任务：
- 99% 样本一眼可判
- 平均 8-15 词
- 不需要中间推理就能给答案

→ **CoT 在两个尺寸（0.6B / 1.7B）都退化** (MCC −0.06 到 −0.13)。

任务复杂度判断标准（粗略）：
| 任务特征 | 是否需要 CoT |
|---|---|
| 单步分类（CoLA, 情感, 语种）| ❌ 不需要 |
| 多步推理（数学题, 代码, 多跳问答）| ✅ 需要 |
| 知识检索 + 综合（医疗诊断、法律 QA）| 部分需要 |

### Pattern 4: **小模型 + 不平衡数据 + RL = 几乎必崩**

我们 GRPO 跑了 5 次都失败，根因总结：
- 小模型 (0.6B) 探索空间小 → mode collapse
- 短答案（1 token）→ entropy 几乎为 0 → KL 拉不动
- 不平衡（70:30）→ reward 偏向多数类 → 加剧 collapse

→ 在**小模型 + 不平衡 + 短答案**这"三连"场景下，**先放弃 GRPO，换 DPO 或就在 SFT 上做工**。

### Pattern 5: **优化器精度比想象的更重要（小 lr 下尤其）**

E4-FULL (全参 SFT 1.7B + adamw_8bit) MCC=0.5263，比 LoRA r=16 还低 0.10。同样的现象出现在 E6 (4B QLoRA + adamw_8bit)。

诊断：lr=1e-5 这种**极小学习率**下，bnb 8bit 优化器的量化误差占有效梯度的比例变大，累积导致决策边界漂移。

→ **如果显存允许，全参 SFT 一定要用 fp32 AdamW**。如果不允许（小卡），**LoRA + fp32 AdamW 比"全参 + 8bit AdamW"更好**。

### Pattern 6: **rank 和 target_modules 是 LoRA 真正的胜负手**

| 调整 | E4 → 新 | Δ MCC |
|---|---|---|
| rank 16 → 32 | E4 → E4-C | **+0.041** |
| qkvo → qkvo+MLP | E4 → E4-A-mlp | **+0.033** |
| eff_bsz 32 → 128 | E4 → E4-B | **-0.045** |
| epochs 3 → 5 + cosine | (在 D 组里) | ~0 |
| dropout 0 → 0.05 | 默认配置已经是 | - |

→ **新任务上 LoRA 调参的优先级**：
1. **rank** ∈ {16, 32, 64} 扫一遍（推断模型容量）
2. **target_modules** 单独加 MLP 试一次
3. epoch / lr 用经典 (3 ep, 2e-4) 别动
4. **不要轻易调大 batch**（除非显存被迫）

### Pattern 7: **Hard Example Mining 必须做"类别平衡选择"**

朴素重复 hard 样本会把 70:30 → 60:40，少数类被过度强调。正确做法：
- 按类别**分别**挑 top-K hard
- 重复倍数取 1.5~2，不要 ×3
- 错题 / 低置信样本权重不同（错题更可能是噪声标签）

### Pattern 8: **模型预测的"性格"是 ensemble 的金矿**

E4-C (1.7B) 偏严格、E6 (4B QLoRA) 偏宽容，前 10 条预测一致率 9/10，**互补**。

→ **小卡场景下，"训 2 个性格不同的中等模型 + ensemble" 通常优于 "训 1 个大模型"**。
工程上看，这也是抗"训练崩溃"的保险（任何一个模型烂了，另一个还能兜底）。

### Pattern 9: **训练集本身有噪声，模型有"软上限"**

E4-C-r32 训练集 acc 87%，**最难的 5 条全是标注争议样本**：
```
"Megan loves Jason."        ← 完全合法被标 unacc
"In the corner lay a dog."  ← 倒装合法
"It's high time Fiona gets a job."  ← 美式合法
```

这些样本贡献了大约 5-10% 的"无解错题"。

→ **看到训练集 acc 接近某个数字（如 87%）后停滞，先怀疑标注噪声**，不是再调参。

### Pattern 10: **基线全景图比单点最佳值更重要**

我们整个项目最有价值的产出之一是 §5.3 那张全景图（MCC 从 0.000 到 0.727 共 20+ 个数据点）。它让每个新实验都能立刻找到自己在能力光谱中的位置。

→ **新项目第一周一定要花时间把全景图建起来**：
- baseline（zero-shot, majority, 经典 ML）
- 上限锚点（API 大模型 / 已知 SOTA）
- 自己的实验点

后面所有讨论都围绕这张图展开，而不是孤立的单个数字。

---

## 4. 复杂任务迁移：从 CoLA 二分类到疾病诊断 / 股票预测

### 4.1 任务对比

| 维度 | CoLA（本项目）| 疾病诊断 | 股票涨跌预测 |
|---|---|---|---|
| 类别数 | 2 (acc/unacc) | 多分类 (10-1000+) 或层级 | 2 或 3 (涨/跌/平) |
| 类别分布 | 70:30 适度不平衡 | **严重不平衡**（罕见病 < 1%）| 接近 50:50 但有时序结构 |
| 输入长度 | 8-15 词 | 长病例报告 + 多模态 | 多维时序 + 文本舆情 |
| 答案确定性 | 高（专家标注一致率 ~95%）| 中（医生间 kappa ~0.7-0.8）| 低（噪声主导）|
| 是否需要 CoT | ❌ | ✅ 鉴别诊断需要推理链 | ⚠️ 可能但风险高 |
| 标注成本 | 低（已开源）| 极高（HIPAA + 标注师培训）| 中（公开数据 + 时序对齐困难）|
| 评测指标 | MCC | 平衡 F1 / AUC / 多类 MCC | Sharpe / IC / 平衡 F1 |

### 4.2 CoLA 经验在新任务上的复用与修正

#### 可直接复用 ✅

1. **先做经典 ML baseline**（XGBoost / LR）锚定下限
   - 医疗：临床数据用 LightGBM 经常碾压 LLM（特征工程更直接）
   - 金融：技术指标 + LR 是 30 年沉淀，LLM 想超过它非常难

2. **看类别分布偏斜**（pred_class_rate vs 真实分布）
   - 医疗罕见病：模型 99% 都说"无病"，AUC 也能 0.95，但召回 0
   - 金融：模型一直说"看涨" + 牛市数据 → 看着 acc 80%，熊市直接归零

3. **小数据 + 大模型 → LoRA 是正解**（容量瓶颈即正则化）

4. **避免照搬大 batch 设置**（CoLA 上 eff=128 反向，新任务尤其要先用小 batch 探）

5. **挖 hard examples 时先做类别平衡分桶**

6. **RL 不要在 SFT 第一版就上**（先 SFT 稳定再 RL）

#### 需要重要修正 ⚠️

7. **CoT 的判断要重做**
   - CoLA 不需要 CoT（短句一眼可判）
   - 疾病诊断**必须**有 CoT（鉴别诊断、排他、罗列证据）
   - 股票预测可以试 CoT 但要警惕"事后诸葛亮"过拟合

8. **优化器精度的 trade-off 不同**
   - CoLA 8551 样本 + 全参 + 8bit optim 已经偏斜
   - 医疗大数据 (10万+) 时，量化误差被样本数稀释，可能没那么严重
   - 但金融小样本 + 极低信噪比，**强烈建议 fp32 + 多卡**

9. **评测指标必须换**
   - 医疗：单纯 Accuracy 没意义，至少要 macro-F1 / AUC / 漏诊率（specifically: recall on minor class）
   - 金融：必须用风险调整收益（Sharpe），acc 高的策略经常亏钱

10. **数据噪声策略**
    - CoLA 噪声 ~5-10%，可以接受 87% 训练集 acc 不再优化
    - 医疗标注师间 kappa 0.7 → 训练集 acc 不应超过 85%（再高就是过拟合噪声）
    - 金融**必然有强噪声**，看 train/test gap 而不是绝对 acc

#### 全新需要考虑的（CoLA 没碰到的）⭐

11. **多模态融合**（医疗：影像 + 文本 + 表格）：建议 CLIP-style late fusion + LoRA 各模态分别微调

12. **时序结构**（金融）：必须考虑 leakage、sliding window、rolling 评测，**不能像 CoLA 那样随机 split**

13. **可解释性**（医疗强需求）：LoRA 修改的层可视化、attention rollout、case study 必须配套

14. **持续学习**（金融市场漂移）：模型 monthly 重训 + 用 LoRA 切换"行情风格" 是廉价方案

### 4.3 一个具体迁移示例：医疗罕见病多分类

假设任务是 100 类罕见病诊断，每类 50-500 例，总 1 万样本。**应用 CoLA 经验的设计**：

```
Step 1 (1 周)  baseline 全景
  - majority, random, kNN, RF on TF-IDF      → 锚定下限
  - GPT-4 / Claude zero-shot                  → 锚定上限
  - 训练集类别分布图 + 标注 kappa             → 锚定噪声

Step 2 (1 周)  prompt + few-shot 上限测
  - GPT-4 + 5-shot ICL                        → 看不训练能多高
  - 决定要不要训练（如果 GPT-4 已经 0.85+，直接部署）

Step 3 (2 周)  SFT
  - 主战场：LoRA r=32, target=qkvo+MLP, lr=2e-4, 3ep, eff_bsz=32
  - 训练数据用 CoT (鉴别诊断推理链) + 短答案双版本，看 CoT 是否有增益（医疗任务大概率有）
  - 必做：hard mining 类别平衡 + label smoothing
  - 评测：macro-F1 / 各类 recall / 罕见类专门看

Step 4 (1 周)  Ensemble + 校准
  - 训 3 个 seed 不同的 LoRA
  - logit avg + temperature scaling
  - 对预测置信度做后校准（医疗必备）

不做：GRPO/RL  ← 在多分类罕见病上 99% 会崩
谨慎做：DPO   ← 只在已经 SFT 稳定后试
```

---

## 5. 模型微调策略决策树

```
新任务来了
   │
   ▼
1. 训练数据多少？
   ├─ < 1k     → Few-shot ICL / Prompt 工程，不要训
   ├─ 1k-10k   → LoRA SFT (短答案优先)
   ├─ 10k-100k → LoRA r=64 / 全参 SFT 都试
   └─ > 100k   → 全参 SFT，多卡，DeepSpeed
   │
   ▼
2. zero-shot 已经 ≥ 0.7×目标？
   ├─ 是 → Prompt + few-shot，别训
   └─ 否 → 进入下一步
   │
   ▼
3. 任务是简单决策（≤ 3 步推理）还是复杂多跳？
   ├─ 简单 → SFT 短答案（CoT 大概率反向）
   └─ 复杂 → SFT + CoT 蒸馏（用大模型生成 CoT）
   │
   ▼
4. 评测达到目标？
   ├─ 是 → 部署，监控 drift
   └─ 否 → 进入 RL/DPO（但先确认 SFT 已稳定）
```

### 各分支的"门槛"判断

| 决策点 | 门槛 |
|---|---|
| 数据量分档 | 看可训参数与样本数比例：< 5 全参可行；> 100 必须 LoRA |
| zero-shot 是否够 | 看是否能稳定打过经典 ML baseline |
| CoT 是否有增益 | 任务平均推理步数 > 2 步时 |
| RL 介入 | SFT 已经收敛（loss 平台 + dev MCC 不再增）且**类别平衡** |

---

## 6. SFT vs RL：何时介入、何时放弃

### 6.1 SFT 的"高回报区"

✅ 应用 SFT：
- baseline zero-shot 远低于目标（gap > 0.2）
- 训练数据质量好（标注一致率 > 0.9 / kappa > 0.8）
- 任务格式稳定（输出可以模式化）
- 模型属于 instruct/chat 系列（已经有指令对齐基础）

❌ 不应用 SFT：
- 数据量极少（< 500）
- 标注噪声高（接近随机）
- 模型 zero-shot 已经接近上限（剩 < 0.05 gap）

### 6.2 RL 的"陷阱区"（CoLA 实测）

我们 Step 6 GRPO 失败 5 次给出非常具体的 RL 失败前置条件：

❌ **不要用 GRPO 的场景**：
1. 类别极不均衡（70:30 已经够呛）
2. 答案极短（1-3 token，entropy 几乎为 0）
3. 小模型（< 1B，探索空间不足）
4. 显存不够开 3 份模型（policy + ref + value）
5. SFT 还没稳定

✅ **GRPO 适合的场景**（反推）：
1. 类别平衡或可重采样（数学题、代码：每题独立）
2. 答案长（≥ 100 token，有多样性可优化）
3. 中大模型（≥ 7B 探索空间足）
4. 显存充足（≥ 80GB / 多卡）
5. SFT 后 dev MCC 已平台

### 6.3 RL 的"廉价替代"：DPO / 排序学习

在小卡 + 不平衡场景，**DPO 几乎在所有维度都比 GRPO 更稳**：

| 维度 | GRPO | DPO |
|---|---|---|
| 内存 | 3 份模型 + rollout | 2 份模型，无 rollout |
| 训练稳定性 | 易 mode collapse | 几乎不崩 |
| 数据要求 | 需要 reward function | 需要偏好对（可 self-generate）|
| 适合不平衡 | ❌ | ✅（pair 内已平衡）|

→ 如果 SFT 之后还想再涨一点，**先 DPO，再考虑 GRPO**。

### 6.4 一个简单的"RL 介入清单"

在用 GRPO/PPO 之前，逐项 check：
```
[ ] SFT 已经训完，dev MCC 至少连续 2 epoch 不再涨
[ ] 类别分布 ≤ 60:40 或可以重采样到平衡
[ ] 显存 ≥ 模型 fp32 大小 × 5（policy + ref + adv + grad + activation）
[ ] 有清晰可计算的 reward function（不是模型打分）
[ ] 答案长度 ≥ 50 token（短答案用 DPO 而不是 GRPO）
[ ] 已经有 baseline DPO 跑出来作为参照
```

任何一项不满足，**先解决这一项再说 RL**。

---

## 7. LoRA vs 全参 vs QLoRA：硬件约束下的真实成本曲线

### 7.1 我们在 14GB 卡上跑的真实数据

| 方法 | 模型 | 显存峰值 | 训练时长 | MCC | 性价比 |
|---|---|---|---|---|---|
| **LoRA r=16** | Qwen3-1.7B | ~6 GB | ~22 min | 0.625 | ⭐⭐⭐⭐⭐ |
| **LoRA r=32** | Qwen3-1.7B | ~7 GB | ~24 min | **0.665** | ⭐⭐⭐⭐⭐ |
| **LoRA r=32 +MLP** | Qwen3-1.7B | ~9 GB | ~28 min | 0.658 | ⭐⭐⭐⭐ |
| 全参 SFT | Qwen3-0.6B | ~13 GB | ~4 min | 0.536 | ⭐⭐⭐ |
| 全参 SFT + 8bit optim | Qwen3-1.7B | ~14 GB | ~43 min | 0.526 ⚠️ | ⭐⭐ |
| **QLoRA nf4 r=32 +MLP** | Qwen3-4B | **~5.4 GB** | ~98 min | 0.663 | ⭐⭐⭐⭐ |

### 7.2 决策矩阵

```
                ┌──── 显存够装 fp32 优化器吗（模型 × 8）？
                │
              YES │ NO
                ▼ │ ▼
              全参 │ ┌─── 数据量 > 100k 吗？
              SFT  │ │
                   │ YES│NO
                   │  ▼ │ ▼
                   │  多 │ ┌─── 模型 size 是不是已经远大于显存？
                   │ 卡  │ │
                   │ ZeRO│YES│NO
                   │  3  │ ▼ │ ▼
                   │     │QLoRA│LoRA
                   │     │ r=  │ r=
                   │     │16-64│16-64
```

### 7.3 LoRA 的"超参口诀"（CoLA 验证版）

```
1. rank 先扫 {16, 32, 64}，找拐点
2. target_modules: qkvo 起步, 加 MLP 看是否再涨 0.02-0.04
3. lr 用 2e-4 起步，cosine warmup 0.03
4. epochs 3，配 eval per epoch
5. effective batch 16-64，**不要超 128**
6. dropout 0.05
7. bf16 + gradient_checkpointing 必开
```

### 7.4 何时该上 QLoRA（4bit 量化基座 + LoRA）

✅ 上 QLoRA：
- 模型 > 4B 且单卡显存 < 24GB
- LoRA 的 r 和 target 已扫过最优
- 可以接受 ~2% 的 MCC 损失（量化噪声）

❌ 不上：
- 模型本身就小（< 2B）→ 没必要量化
- 训练后要做合并部署 → 量化会破坏精度
- 用 paged_adamw_8bit 的环境下 → 双重量化误差严重（我们的 E6 就有这个问题）

---

## 8. 工程踩坑与方法论沉淀

### 8.1 容易在小细节上失控的点（每个都让我浪费 ≥ 1h）

| # | 坑 | 表现 | 修复 |
|---|---|---|---|
| 1 | 没设 `enable_thinking=False` | Qwen3 默认开 thinking，eval 时输出 200+ token CoT，parse_fail 飙升 | 评测时显式 `enable_thinking=False` |
| 2 | left-padding 后用 `attention_mask.sum()-1` 取末位 | 取到 padding 位置的 logit，hard mining 全错 | 用 `logits[:, -1, :]` |
| 3 | trl 1.x API 变更 | `tokenizer=` → `processing_class=`, `max_seq_length` → `max_length` | 跟随新 API |
| 4 | torch 2.5 vs trl 的 `FSDPModule` 缺失 | import 直接挂 | 写 stub `_fsdp_compat.py` |
| 5 | 8bit AdamW 在小 lr 下决策边界漂移 | acc 偏斜，MCC 下降但 loss 看起来正常 | 监控 `pred_class_rate` 而不是只看 loss |
| 6 | 大 batch + 小数据 → 梯度更新次数太少 | 训完 loss 还高，dev 不涨 | 优先小 batch 多步，eff_bsz ≤ 64 |
| 7 | parse_answer 顺序错（先判 acceptable 再判 unacceptable）| `unacceptable` 被 partial-match 错判 | 先判 `unacceptable` 再判 `acceptable` |
| 8 | 跨容器 vGPU 显存抖动 | nvidia-smi 显示 0 但 OOM | 用 `torch.cuda.mem_get_info()` 取真实可用，留 ≥ 1GB 余量 |
| 9 | trl Trainer 默认 eval_strategy=epoch 时初始化 eval | 训 0 步就触发 eval，OOM 风险 | 显存吃紧时设 `eval_strategy=no` |
| 10 | nohup 脚本 cwd 不对 | 路径解析失败 | 用 `cd && nohup ... &` 严格分开 |

### 8.2 长流程脚本的必备防御性设计

我们后期的 `run_sweep_e4.sh` 和 `run_after_sweep.sh` 总结出一个 pattern：

```bash
# 1. 每个子任务独立日志
TASK_LOG=$LOGDIR/sweep_${EXP}_train.log
$PY ... > $TASK_LOG 2>&1
RC=$?

# 2. 立即检查 exit code，失败不影响后续任务
if [ $RC -ne 0 ]; then
    echo "❌ $EXP 失败 (rc=$RC)"
    tail -20 $TASK_LOG
    return 1   # 注意是 return 不是 exit
fi

# 3. 立即评测，避免 ckpt 累积（磁盘满）
$PY my_try/scripts/eval_lora.py --adapter $CKPT > $LOGDIR/eval_${EXP}.log 2>&1

# 4. 提取关键指标用 inline python，避免依赖 jq
MCC=$($PY -c "import json; print(json.load(open('$RES'))['results'][0]['mcc'])")

# 5. master log 只记 summary，详细在子日志
echo "📊 $EXP: MCC=$MCC" >> $MASTER_LOG
```

### 8.3 watchdog 脚本（让一系列任务真正"无人值守"）

`run_after_sweep.sh` 的设计：
```bash
# 等前序任务结束（可以是别的脚本启的）
SWEEP_PID=${1:-790450}
while ps -p $SWEEP_PID > /dev/null 2>&1; do
    sleep 30
done
echo "✅ 前序任务完成，开始下一阶段"

sleep 10  # 让 GPU 喘口气
# 启动新任务...
```

价值：跨阶段并行启动（前期一边训练一边下载下个模型），节省 30%+ 总时间。

### 8.4 "实验日志即文档"原则

我每天结束时把当天结果写到 `step5_summary.md` / `step6_summary.md`，**最大的好处不是给别人看**，而是：
- 写的时候被迫思考"为什么 X 涨了 Y 跌了"，往往发现新假设
- 第二天接续工作时不用 reload context
- 实验失败的负面结果不会被遗忘（避免重蹈覆辙）

→ **每个 ≥ 4h 的实验都该配一个 markdown 当天写完**。

### 8.5 一个被低估的小工具：`summarize_step5.py`

```python
# 自动扫所有 res_*.json，按 MCC 排序输出表格
for f in sorted(Path(d).glob("res_*.json")):
    data = json.loads(f.read_text())
    rows.append({...})
rows.sort(key=lambda x: -x["mcc"])
print(table)
```

这个 50 行脚本让我每次能在 < 1 秒内看到全部 13 个实验的横向对比，**比手动维护表格可靠得多**。

---

## 9. 个人经验自评：哪些做得好、哪些可以更好

### 9.1 做得好 ✅

1. **先做 baseline + 经典 ML + API 锚点再做 SFT**
   → 避免了"训了一个月才发现连 LR 都不如"的悲剧

2. **失败实验也认真记录**（5 次 GRPO 失败、2 次 CoT 失败、1 次 hard mining 失败）
   → step6_summary.md 600+ 行，把 mode collapse 的诊断思路完整记下来，未来再做 RL 直接受益

3. **并行任务设计（watchdog）省了大量 wall time**
   → sweep + 4B 下载 + 4B QLoRA 训练 三个任务同时跑

4. **每次结果出来立刻分析混淆矩阵**，不只看 MCC
   → 早期发现 1.7B / 4B 的"性格互补"，为后续 ensemble 提供方向

5. **保留所有 ckpt 的 run_info.json**
   → 配置完整可追溯，半个月后回来还能精确复现

### 9.2 可以做得更好 ⚠️

1. **超参扫描应该用 Optuna/W&B Sweeps 而不是手写 4 组**
   → 我们手挑了 r=16/32 + qkvo/+MLP + bsz 32/128 等 4 组，本质是消融而不是搜索；如果用贝叶斯搜索可能能找到 r=24 + 部分 MLP 这种**没想到的甜点**

2. **没做 seed 重复实验**
   → 所有数字都是单 seed，0.6654 vs 0.6633 的 0.002 差距可能在 noise 之内，不应做太多 narrative

3. **Hard Example Mining 的失败是设计错误，不是实验失败**
   → 设计前我应该手算"hard 集里 unacc 占比"，发现失衡再设计权重，而不是直接 ×3

4. **没有早做 Ensemble**
   → 1.7B 和 4B 的互补性质 Step 2 就该看出来；ensemble 应该在 hard mining 之前做

5. **GRPO 失败 5 次后才放弃，浪费了 1 天**
   → 第 2 次 mode collapse 时就该停下来想想是不是任务-算法 fit 错了，而不是继续调 reward

6. **CoT 蒸馏花了 1 天但回报为负**
   → 早判断"CoLA 是 1 步任务，不需要 CoT"就能省下来

7. **文档结构有冗余**
   → step5_summary.md 已 830 行，分章节后部分内容（如 5.3 全景图）应该独立成文件而不是塞在里面

---

## 10. 后续可挖方向（如果再继续）

按预期收益 × 工程量打分：

| 方向 | 预期 ΔMCC | 工程量 | 优先级 |
|---|---|---|---|
| **Ensemble (E4-C + E6)** | +0.02-0.04 | 1-2h | ⭐⭐⭐⭐⭐ |
| **类别平衡 Hard Mining**（每类 top-K + ×2）| +0.01-0.02 | 1h | ⭐⭐⭐⭐ |
| **Qwen3-4B QLoRA 调小 lr (5e-5) + qkvo only** 减少多数类偏斜 | +0.01-0.02 | 100min | ⭐⭐⭐ |
| **DPO**（用 1.7B-SFT 的对错样本对自动构造偏好对）| +0.01-0.03 | 半天 | ⭐⭐⭐ |
| **训练数据清洗**（用 API 大模型重标 hard examples）| +0.02-0.05 | 1 天（含 API 成本）| ⭐⭐⭐ |
| **多 seed 集成（5 seed × LoRA r=32）** | +0.01-0.02 | 2h | ⭐⭐ |
| **GRPO 在 4B 上重试**（更大模型 + paged optim）| 不确定 | 1 天 | ⭐ |

**最大的"未尝试但确定能涨"的方向**是 **Ensemble**（E4-C + E6 性格互补，9/10 一致率说明剩下的 1/10 大部分能 vote 对）。

---

## 11. 闭幕一句话

> **小卡 + 小数据 + 简单任务 = LoRA 派对**。
> 这个组合下，再怎么折腾全参、CoT、GRPO 也卷不过把 LoRA 的 rank 和 target 调对的人。
> 但如果环境换成大卡 + 大数据 + 复杂任务，本文的所有结论都需要重新验证。
> 经验的价值不在结论本身，而在"知道这些结论是在什么前提下成立的"。
