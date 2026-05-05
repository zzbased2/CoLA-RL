# CoLA-RL Step 4：不训练对比校验汇总

> 创建日期：2026-05-04
> 脚本：`my_try/eval_baseline.py`（LLM） + `my_try/eval_classical.py`（经典 ML）
> 结果 JSON：`my_try/baseline_results.json` + `my_try/res_*.json` + `my_try/baseline_classical.json`
> 对应文档：`my_try.md §4`
> 前置：Step 3 Baseline 评测（zero-shot 非思考）

---

## 1. 本轮目标

在**完全不训练**的前提下，把四个模型在 CoLA 验证集（527 条）上用不同"免训练"手段拉一遍，回答两个问题：

1. **prompt 工程 / 推理模式能把小模型拉到多高？** → 决定"SFT 是否真的必要"
2. **非 LLM 经典方法的"下限"是多少？** → LLM 至少要比 TF-IDF + LR 更强才值得部署

---

## 2. 实验矩阵

### 2.1 LLM：4 模型 × 4 变体 = 16 组

| 变体 | 说明 | `enable_thinking` | `max_new_tokens` |
| --- | --- | --- | --- |
| `plain`（=Step 3 zero-shot） | 官方 prompt，直接问 | False | 32 |
| `fewshot` | plain prompt 前置 **4 条示例**（2 正 2 负，in-context） | False | 32 |
| `cot` | plain prompt 末尾追加 `"Let's think step by step."` | False | 256 |
| `thinking` | plain prompt + **打开 Qwen3 原生 `<think>` 模式** | True | 1024 |

### 2.2 经典 ML：5 个非 LLM baseline（全部 CPU，总耗时 <5s）

| ID | 方法 | 思路 |
| --- | --- | --- |
| B1 | Majority | 全预测 `acceptable`（训练集多数类） |
| B2 | Random | 按训练集先验（70.4% acc）抽签 |
| B3 | LengthThresh | 句子 token 数 > k 判为 acceptable，k 在训练集上搜最优 |
| B4 | CharNgramLR | char 2-5 gram + TF-IDF + LogisticRegression |
| B5 | WordNgramLR | word 1-2 gram + TF-IDF + LogisticRegression |

### 2.3 API 大模型（补测，通过 CodeBuddy 开放平台）

本地 14 GB L20 跑不了 32B+ 的模型，但通过 `copilot.tencent.com/v2/chat/completions`
可以调用闭源/超大模型做同口径 zero-shot 评测，作为"能力天花板"参考。

| 模型 | 平台路由 | 定位 | 脚本 |
| :--- | :--- | :--- | :--- |
| `hy3-preview` | 腾讯混元 3.0（推理型）| 国产 thinking 模型 | `my_try/eval_api_models.py` |
| `deepseek-v3-0324` | DeepSeek V3（0324 快照）| 通用对话 | 同上 |
| `claude-opus-4.7` | Anthropic 旗舰 | 英语语法任务公认最强 | 同上 |
| `gemini-3.1-flash-lite` | Google Gemini 3.1 轻量 | 低延迟对话 | 同上 |

Prompt 与本地完全一致（PLAIN_TEMPLATE），max_new_tokens=16（hy3-preview 因思维链太长单独设 1024）。

---

## 3. 核心结果表（MCC 为主指标）

### 3.1 LLM 全矩阵

| 模型 | plain | fewshot | cot | thinking | 最佳 |
| :--- | :-: | :-: | :-: | :-: | :-: |
| **Qwen3-0.6B** | 0.000 | 0.000 | 0.000 | **0.254** | thinking |
| **Qwen3-1.7B** | **0.500** | 0.499 | 0.139 | 0.431 | plain |
| **Qwen3.5-0.8B** | **0.340** | 0.192 | 0.203 | 0.211 | plain |
| **Qwen3.5-2B** | **0.549** | 0.498 | 0.214 | 0.483 | plain |

> 粗体 = 每行最佳 MCC。

### 3.2 经典 baseline

| 方法 | MCC | Accuracy | 备注 |
| :--- | :-: | :-: | :--- |
| B1 Majority | 0.000 | 0.693 | 和 Qwen3-0.6B zero-shot 一样（都是"全说 yes"） |
| B2 Random | −0.027 | 0.569 | 基本等于瞎猜 |
| B3 LengthThresh | 0.078 | 0.687 | 最优阈值 k=4，说明长度信号非常弱 |
| B4 CharNgramLR | 0.129 | 0.600 | 26632 特征 |
| B5 WordNgramLR | **0.176** | 0.636 | 12267 特征，经典 ML 的最高水位 |

### 3.3 API 大模型（zero-shot plain）

| 模型 | **MCC** | Accuracy | Parse fail | 耗时/单条 | 备注 |
| :--- | :-: | :-: | :-: | :-: | :--- |
| **deepseek-v3-0324** | **0.7268** | 0.882 | 0 | 2.2 s | 与原项目 README 报告 0.726 完美对齐 ✅ |
| **claude-opus-4.7** | **0.7180** | 0.879 | 0 | 4.1 s | Anthropic 旗舰，语法任务强 |
| **gemini-3.1-flash-lite** | **0.7030** | 0.873 | 0 | 2.2 s | Google 轻量版也能打 |
| **hy3-preview** | 0.5274 | 0.797 | 15 | 7.3 s | 国产 thinking 模型，**反而更差** |

**观察**：
- 三个**通用大模型 MCC 都在 0.70+**，远高于本地 Qwen3 系列 baseline 的 0.50-0.55
- **hy3-preview 作为推理型模型，在 CoLA 二分类上表现最差**（和 Step 4 §4.3 结论一致：CoLA 不需要 reasoning）
- 这三个 0.70+ 的模型确定了我们的"**能力天花板**"：Qwen3-0.6B 怎么训也很难突破 0.70（除非蒸馏 CoT）

### 3.4 全方法并列排序（MCC 降序）

| 排名 | 方法 | MCC | 类型 | 耗时 |
| :-: | :--- | :-: | :-- | :-: |
| 1 | **deepseek-v3-0324** (API) | **0.727** | API 大模型 | 1173s |
| 2 | **claude-opus-4.7** (API) | **0.718** | API 大模型 | 2161s |
| 3 | **gemini-3.1-flash-lite** (API) | **0.703** | API 大模型 | 1171s |
| 4 | Qwen3.5-2B (plain) | **0.549** | 2B LLM | 140s |
| 5 | hy3-preview (API) | 0.527 | API thinking 模型 | 3834s |
| 6 | Qwen3.5-2B (thinking) | 0.483 | 2B LLM + thinking | 5531s |
| 7 | Qwen3.5-2B (fewshot) | 0.498 | 2B LLM | 34s |
| 8 | Qwen3-1.7B (plain) | 0.500 | 1.7B LLM | 62s |
| 9 | Qwen3-1.7B (fewshot) | 0.499 | 1.7B LLM | 20s |
| 10 | Qwen3-1.7B (thinking) | 0.431 | 1.7B LLM + thinking | 2612s |
| 11 | Qwen3.5-0.8B (plain) | 0.340 | 0.8B LLM | 114s |
| 12 | Qwen3-0.6B (thinking) | 0.254 | 0.6B LLM + thinking | 992s |
| 13 | Qwen3.5-2B (cot) | 0.214 | 2B LLM + CoT | 775s |
| 14 | Qwen3.5-0.8B (thinking) | 0.211 | 0.8B + thinking | 2398s |
| 15 | Qwen3.5-0.8B (cot) | 0.203 | 0.8B + CoT | 414s |
| 16 | Qwen3.5-0.8B (fewshot) | 0.192 | 0.8B LLM | 26s |
| 17 | **B5 WordNgramLR** | **0.176** | **经典 ML** | **2s** |
| 18 | Qwen3-1.7B (cot) | 0.139 | 1.7B + CoT | 711s |
| 19 | B4 CharNgramLR | 0.129 | 经典 ML | 3s |
| 20 | B3 LengthThresh | 0.078 | 启发式 | <1s |
| 21 | Qwen3-0.6B (plain/fewshot/cot) | 0.000 | 0.6B LLM | - |
| 22 | B1 Majority | 0.000 | 启发式 | - |
| 23 | B2 Random | −0.027 | 启发式 | - |

---

## 4. 六个关键发现

### 4.0 API 大模型定义了能力天花板，推理型模型反而偏弱

三个通用大模型（deepseek-v3-0324 / claude-opus-4.7 / gemini-3.1-flash-lite）的 MCC 稳定在 **0.70-0.73**，比本地最强的 Qwen3.5-2B zero-shot (0.549) 高 **+0.16 左右**。

值得注意的是：
- **hy3-preview 推理型模型 MCC 只有 0.527**，比同平台的 deepseek 低了 0.20
- 解析失败 15 条（其他三家都 0），证明 thinking 类模型**输出格式不稳定**
- 再次印证 Step 4 §4.3 的结论：**CoLA 这种"秒级二分类"任务不需要 reasoning**

**锚点意义**：这 3 个 API 模型把我们的能力天花板钉在 **0.73**。0.6B/1.7B 系列通过 LoRA SFT 能达到多少，实际上就是对比"**多大的模型+多轻的训练=多少 MCC**"的性价比。

---

### 4.1 Qwen3-0.6B 在普通模式下 MCC = 0.000，等同"全说 yes"

- plain / fewshot / cot 三种模式 MCC 全部为 0
- 对所有 527 条样本都输出 `acceptable`（见混淆矩阵右列全为 1）
- **这不是 prompt 写得不好，而是 0.6B 模型的"能力天花板"**
- 只有打开 thinking（原生 `<think>` 推理）才能拉到 0.254
- 结论：**0.6B 是 SFT 必要性最强的证据**，baseline 0.000 给 SFT 留下巨大空间

### 4.2 Prompt 工程基本没有正收益，甚至负收益

对比每个模型的"最佳非 plain 变体"相对 plain 的 Δ：

| 模型 | plain | 最佳非 plain | Δ |
| :--- | :-: | :-: | :-: |
| Qwen3-0.6B | 0.000 | 0.254 (thinking) | **+0.254** |
| Qwen3-1.7B | 0.500 | 0.499 (fewshot) | **−0.001** |
| Qwen3.5-0.8B | 0.340 | 0.211 (thinking) | **−0.129** |
| Qwen3.5-2B | 0.549 | 0.498 (fewshot) | **−0.051** |

**只有 0.6B 能从 prompt 变体中获益，其它三个模型 plain 就是最强。**

原因推测：
- 1.7B / 2B 已经掌握了这个任务，追加的推理约束（think / CoT）反而是噪声
- CoT 诱导尤其差：把 1.7B 从 0.50 打到 0.14，失败 37 条；把 2B 从 0.55 打到 0.21，失败 63 条
- **给擅长任务的模型强行加"思考"，反而让它漂移到非典型输出，解析失败率陡升**

### 4.3 Thinking 模式的成本-收益极差（0.6B 例外）

| 模型 | thinking MCC | thinking 耗时 | 每 +0.001 MCC 付出秒数 |
| :--- | :-: | :-: | :-: |
| Qwen3-0.6B | 0.254 (Δ=+0.254) | 992s | 3.9 s/Δ‰ ✅ 值得 |
| Qwen3-1.7B | 0.431 (Δ=−0.069) | 2612s | ✗ 负收益还贵 |
| Qwen3.5-0.8B | 0.211 (Δ=−0.129) | 2398s | ✗ 负收益还贵 |
| Qwen3.5-2B | 0.483 (Δ=−0.066) | 5531s | ✗ 负收益还贵（耗时将近 92 分钟） |

- thinking 平均生成 200-800 token，比 plain 贵 50-400 倍
- 对 0.6B 是救命稻草，对其他模型是赔本买卖
- 工程启示：**CoLA 这种简单二分类任务根本不需要 reasoning；Qwen3 的 thinking 能力是为数学/代码准备的**

### 4.4 LLM 必须跑到 1.7B 才稳压经典 ML

WordNgramLR（MCC=0.176）这条**耗时 2 秒的线**非常关键：
- Qwen3-0.6B 所有变体（除 thinking=0.254 外）全部 **≤ 0.000**，**打不过 LR**
- Qwen3.5-0.8B plain（0.340）能打过 LR，但 fewshot/cot/thinking 都跌到 0.19-0.21，**刚好在 LR 附近**
- Qwen3-1.7B 和 Qwen3.5-2B 才稳定甩开 LR 一倍以上

**反直觉结论**：用户量大、延迟敏感的场景下，如果只能用 0.6B 或没有推理加速，**TF-IDF + LR 可能是更好的选择**；除非你愿意上 1.7B+。

### 4.5 Qwen3 vs Qwen3.5 的"架构代差"仍被"参数量差距"压倒

| 对比 | Qwen3 | Qwen3.5 | 谁赢 |
| :--- | :-: | :-: | :-: |
| Qwen3-0.6B plain vs Qwen3.5-0.8B plain | 0.000 | 0.340 | 3.5-0.8B |
| Qwen3-1.7B plain vs Qwen3.5-2B plain | 0.500 | 0.549 | 3.5-2B（微弱） |
| Qwen3-1.7B plain vs Qwen3.5-0.8B plain | **0.500** | 0.340 | **Qwen3-1.7B** |

- 同代更大参数 > 新代更小参数（1.7B > 0.8B）
- 同规模新代 > 同规模旧代（0.8B > 0.6B）
- **参数量的增益比"代际升级"大很多**
- 对 SFT 的启示：不要为了"追新版本"换 0.8B/2B 这种奇怪档位，**标准 Qwen3-0.6B/1.7B 的架构对 peft/trl 更友好**，是主战场

---

## 5. 混淆矩阵详细对比（谁在瞎说 yes）

数据分布：acceptable=365 (69.3%)，unacceptable=162 (30.7%)

### 5.1 每个最佳变体的预测倾向

| 方法 | pred_acc_rate | 真实 0.693 | 偏差 | 风格 |
| :--- | :-: | :-: | :-: | :--- |
| Qwen3-0.6B plain | 1.000 | | +0.307 | **全说 yes**（等于 Majority） |
| Qwen3-0.6B thinking | 0.879 | | +0.186 | 偏乐观 |
| Qwen3-1.7B plain | 0.715 | | +0.022 | ✅ 最接近真实分布 |
| Qwen3.5-0.8B plain | 0.820 | | +0.127 | 偏乐观 |
| Qwen3.5-2B plain | 0.596 | | −0.097 | 偏保守 |
| B5 WordNgramLR | 0.651 | | −0.042 | 稍偏保守 |

### 5.2 观察

- **所有 LLM 都偏"说 yes"**（除 Qwen3.5-2B plain 稍偏 no）
- Qwen3-1.7B plain 的预测分布最均衡（pred_acc=71.5% vs 真实 69.3%），这也是它 MCC 这么高的原因
- **0.6B 的"全说 yes"行为**不是 prompt 没写清楚，而是模型在二分类边界上完全无信号——SFT 必须注入"unacceptable"的偏好

---

## 6. 推理速度 vs 模型规模

| 方法 | it/s (plain) | 对应每千条预估 |
| :--- | :-: | :-: |
| Qwen3-0.6B plain | 12.05 | 1.4 min |
| Qwen3-1.7B plain | 8.47 | 2.0 min |
| Qwen3.5-0.8B plain | 4.62 | 3.6 min |
| Qwen3.5-2B plain | 3.75 | 4.4 min |
| B5 WordNgramLR（CPU） | ~260 | 3.8 s |

- Qwen3.5 系列（0.8B/2B）比对应规模 Qwen3 慢一倍多，因为混合注意力（Gated DeltaNet + Gated Attention）和视觉分支走的是非优化路径
- **在 Qwen3.5 上做 SFT 的性价比低**；Step 5 优先标准 Qwen3

---

## 7. 给 Step 5 LoRA SFT 的 4 条设计原则

基于本轮所有数据：

1. **主攻 Qwen3-0.6B**
   - baseline 0.000，SFT 目标 ≥0.40
   - 0.40 = 超越自身 thinking 的 0.254，逼近 1.7B 的 zero-shot 0.500
   - 这是最能体现 LoRA-SFT 价值的对象

2. **1.7B 做对照**
   - baseline 已 0.500，目标 ≥0.62（接近原项目全参 SFT 报告的 0.657）
   - 看"已经会了的模型"能被 LoRA 再拉多少

3. **Qwen3.5 系列降权**
   - 0.8B baseline 弱（0.34），架构非标，LoRA target_modules 还要探
   - 2B baseline 强（0.55）但推理慢、显存吃紧
   - 选其一做 QLoRA 对比即可，**优先 Qwen3.5-2B**（baseline 最强）

4. **不要让模型思考 / CoT**
   - CoT 数据会让模型输出漂移，parse_fail 飙升
   - SFT 数据就用 **纯 `acceptable`/`unacceptable` 短答案**（原项目方案 A）
   - 如果后面真想做 CoT，等到有 RL 资源 + R1 蒸馏数据再说

---

## 8. 产出物清单

| 文件 | 作用 |
| :--- | :--- |
| `my_try/eval_baseline.py` | LLM 评测脚本（支持 plain/fewshot/cot/thinking 四种 prompt 变体） |
| `my_try/eval_classical.py` | 经典 ML baseline 脚本（5 个方法，CPU） |
| `my_try/eval_api_models.py` | **API 大模型评测脚本**（CodeBuddy 平台，流式） |
| `my_try/baseline_results.json` | Step 3 zero-shot 4 模型结果（plain 变体） |
| `my_try/baseline_classical.json` | 5 个经典 baseline 结果 |
| `my_try/res_{model}_{variant}.json` × 12 | Step 4 各个变体的细节（含前 10 条样本 raw output） |
| **`my_try/res_api_{model}.json` × 4** | **API 模型 zero-shot 结果**（deepseek / claude / gemini / hy3） |
| `my_try/baseline_summary.md` | Step 3 汇总（zero-shot） |
| `my_try/step4_summary.md` | 本文档（Step 4 完整汇总） |

---

## 9. 总结一句话

> 本地 4 模型在 CoLA 上的 **免训练上限是 0.549**（Qwen3.5-2B plain），
> API 大模型的天花板是 **0.73**（deepseek-v3-0324），
> 而 Qwen3-0.6B 在 plain 下 **MCC=0**。
> Prompt 工程（fewshot/cot/thinking）**不能打破规模瓶颈**，甚至会给强模型带来负收益。
> **推理型模型（hy3-preview）在这种简单二分类任务上反而最差**——再次证明 reasoning 不是万能钥匙。
> 所以 **Step 5 LoRA SFT 的动机完全成立**，尤其是为 Qwen3-0.6B 注入"unacceptable"的判别能力。
