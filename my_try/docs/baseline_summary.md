# CoLA-RL Baseline 评测结果（Step 3）

> 脚本：`my_try/eval_baseline.py`
> 结果 JSON：`my_try/baseline_results.json`
> 评测日期：2026-05-04
> 硬件：NVIDIA L20 × 1（15 GB，bf16）

---

## 1. 实验设置

| 项 | 值 |
| --- | --- |
| 验证集 | `cola_data/in_domain_dev.tsv`（527 条） |
| 标签分布 | acceptable=365（69.3%）/ unacceptable=162（30.7%） |
| Prompt | `my_try.md §3.2` 的官方 prompt（不含示例，zero-shot） |
| 生成方式 | Greedy（`do_sample=False`） |
| `max_new_tokens` | 32 |
| `enable_thinking` | **False**（快速版，文档的 thinking 版预计耗时 ×5-10，暂未跑） |
| 精度 | bf16 |
| 评测器 | `sklearn.metrics.matthews_corrcoef` |

## 2. 核心结果

| 模型 | 参数 | **MCC** | Accuracy | ParseFail | 耗时 | 吞吐 (it/s) |
| --- | --- | --- | --- | --- | --- | --- |
| **Qwen3-0.6B** | 0.6B | **0.0000** | 0.6926 | 0 / 527 | 43.7s | 12.05 |
| **Qwen3-1.7B** | 1.7B | **0.5002** | 0.7913 | 0 / 527 | 62.2s | 8.47 |
| **Qwen3.5-0.8B** | 0.8B | **0.3402** | 0.7438 | 0 / 527 | 114.1s | 4.62 |
| **Qwen3.5-2B** | 2B | **0.5491** | 0.7856 | 0 / 527 | 140.5s | 3.75 |

合计耗时：**6 分 00 秒**（比 `my_try.md` 预估的 1 小时快得多）

## 3. 混淆矩阵（行=真实，列=预测；标签顺序 [unacc, acc]）

```
                       pred=unacc  pred=acc
Qwen3-0.6B
  true=unacc                 0        162     ← 162 条 unacc 全部预测错
  true=acc                   0        365
  → 退化为"全部输出 acceptable"

Qwen3-1.7B
  true=unacc               101         61     ← 62% unacc 召回
  true=acc                  49        316     ← 86% acc 召回
  → 平衡最好的一个

Qwen3.5-0.8B
  true=unacc                61        101     ← 38% unacc 召回（偏乐观）
  true=acc                  34        331     ← 91% acc 召回

Qwen3.5-2B
  true=unacc               131         31     ← 81% unacc 召回（最好）
  true=acc                  82        283     ← 77% acc 召回（偏保守）
```

**预测 acceptable 的比例**（真实值 0.693）：

| 模型 | pred_acc_rate | 偏差 |
| --- | --- | --- |
| Qwen3-0.6B | 1.000 | **完全倾向 yes** |
| Qwen3-1.7B | 0.715 | +0.022（几乎持平真实分布） |
| Qwen3.5-0.8B | 0.820 | +0.127（偏乐观） |
| Qwen3.5-2B | 0.596 | −0.097（偏保守） |

## 4. 关键结论

### 4.1 为什么要做 SFT：Qwen3-0.6B 的 MCC=0 是最强动机

Qwen3-0.6B 在 zero-shot 非思考模式下**完全不具备 CoLA 判别能力**——
无论句子是否语法正确，它都输出 `acceptable`。这意味着：
- 小模型即使经过大规模预训练，**也不会自动掌握"合格/不合格"这种二分类偏好**
- 0.6B baseline 的 MCC = 0.0 给 SFT 留下了**巨大的提升空间**
- Step 5 LoRA SFT 的目标很明确：**从 0.0 把 0.6B 拉到 ≥0.3**（达到官方 0.223 的 MCC 即算"追上了 thinking 版的表现"）

### 4.2 规模效应 & 架构代差

- **Qwen3 系列内部**：0.6B (0.00) → 1.7B (0.50) 跳跃极大，说明 1.7B 刚好过了"掌握这个任务"的门槛
- **Qwen3.5 系列内部**：0.8B (0.34) → 2B (0.55)，也有显著增益
- **同代跨规模**：Qwen3-1.7B (0.50) 和 Qwen3.5-2B (0.55) 相当，2B 略优
- **跨代对比**：Qwen3.5-0.8B (0.34) 显著**弱于** Qwen3-1.7B (0.50)，说明**参数量比"架构代数"更重要**

### 4.3 Parse 成功率 100%

所有模型在 `max_new_tokens=32` 的硬限下都能直接输出单词 `acceptable` / `unacceptable`，
**prompt 设计完全不需要改**。甚至大部分样本是 "1 token" 长度的输出，
所以思考模式带来的额外 token 预算其实对非 thinking 模型无意义。

### 4.4 性能实测 vs 预估

文档预估每模型 10-20 分钟，实际：

| 模型 | 文档预估 | 实际 | 比例 |
| --- | --- | --- | --- |
| 0.6B | 10-20 min | 43.7s | ≈ 15× 快 |
| 1.7B | 10-20 min | 62.2s | ≈ 12× 快 |
| 0.8B | 10-20 min | 114.1s | ≈ 6× 快 |
| 2B | 10-20 min | 140.5s | ≈ 5× 快 |

原因：
1. L20 SM count 92，算力比文档设想的 T4/3090 档要强
2. 每个样本生成 token 数极少（大多 1 个 token 就结束），实际负担比 `max_new_tokens=32` 小很多
3. bf16 比 fp32 快 ~2×

→ **意味着 thinking 模式（max_new_tokens=1024）跑全量也只要 ~1 小时**，后续有时间时可以加跑。

## 5. 后续行动建议

根据 baseline 结论，我建议 Step 4 / Step 5 调整如下（供用户决策）：

### Step 4：不训练对比校验的主角 = Qwen3-0.6B

原计划是"用不同 prompt / few-shot 看最高能到多少，作为 SFT 必要性判断"。
鉴于 **0.6B 在 zero-shot 下 MCC=0，已经证明了 SFT 的必要性**，Step 4 可以：

- **简化方案**：只在 0.6B 上跑 2-3 个变体（few-shot 3 条 / thinking 模式 / CoT prompt），
  看 prompt 工程能把 0.6B 从 0.0 拉多高。如果 prompt 能轻松拉到 0.4+，SFT 价值就降低了；
  如果 prompt 怎么改都低于 0.2，则 SFT 是必须的。
- 预计耗时：3 个 prompt × 0.6B × zero-shot ≈ 10 分钟内搞定

### Step 5：LoRA SFT 的重点目标模型 = Qwen3-0.6B

- **主攻 0.6B**：baseline 0.00，SFT 后目标 ≥0.40（达到甚至超越 1.7B 的 baseline）
- **陪跑 1.7B 或 2B**：baseline 已达 0.50+，看 SFT 能否推到 0.60+，作为"大模型 SFT 上限"参考
- **0.8B 跳过**：架构代差问题，即使 SFT 后也大概率弱于同规模的 Qwen3 系列

### （暂缓）Thinking 模式评测

是否要跑 `--thinking --max_new_tokens 1024`？
- **建议只对 0.6B 跑一次**，看 thinking 能把"全说 yes"修正到什么水平
- 如果 thinking 后 0.6B MCC 能到 0.2+（匹配官方报告），那 SFT 的"净增益"需要扣掉 thinking 的贡献再算

## 6. 产出物清单

| 文件 | 作用 |
| --- | --- |
| `my_try/eval_baseline.py` | 评测脚本（已修 transformers 5.x API 坑，可复用于 Step 5 SFT 后评测） |
| `my_try/baseline_results.json` | 4 模型完整结果（MCC / 混淆矩阵 / 首 10 条样本 raw_output） |
| `my_try/baseline_summary.md` | 本文档 |
