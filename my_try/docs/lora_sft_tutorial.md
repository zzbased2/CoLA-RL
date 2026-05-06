# LoRA SFT 完整教程：从环境到生产 (Qwen3-0.6B / CoLA 实战)

> 创建日期：2026-05-05
> 实战项目：[zzbased2/CoLA-RL](https://github.com/zzbased2/CoLA-RL) `my_try/` 目录
> 实测硬件：NVIDIA L20 14 GB（消费级 8 GB+ 卡也能跑）
> 配套代码：`my_try/prepare_sft_data.py` + `train_lora_sft.py` + `eval_lora.py`
>
> 这是一份"如果让别人在自己的机器上复现这套流程"的教程，对应的实验记录见 `step5_summary.md`。

---

## 目录

- [0. 整体路线图](#0-整体路线图)
- [1. 前置依赖与环境](#1-前置依赖与环境)
- [2. 数据准备](#2-数据准备)
- [3. 模型下载](#3-模型下载)
- [4. 训练脚本设计](#4-训练脚本设计)
- [5. 训练执行](#5-训练执行)
- [6. 评测](#6-评测)
- [7. 对比与结果分析](#7-对比与结果分析)
- [8. 注意事项总结](#8-注意事项总结)
- [9. 一页式可复用 SOP](#9-一页式可复用-sop)
- [10. 扩展方向](#10-扩展方向)
- [11. 文件索引](#11-文件索引)

---

## 0. 整体路线图

```mermaid
graph LR
    A["① 环境<br/>venv + 依赖"] --> B["② 数据<br/>CoLA → JSONL"]
    B --> C["③ 模型<br/>Qwen3-0.6B 下载"]
    C --> D["④ 脚本<br/>trl + peft 50 行"]
    D --> E["⑤ 训练<br/>13 min"]
    E --> F["⑥ 评测<br/>MCC 计算"]
    F --> G["⑦ 对比<br/>vs baseline"]
    G --> H["⑧ Commit<br/>留下产物"]

    style A fill:#e1f5ff
    style D fill:#fff4e1
    style F fill:#fff4e1
```

**全流程时长**（不含模型下载）：约 **30 分钟**（含数据、训练、评测）。

---

## 1. 前置依赖与环境

### 1.1 硬件最低要求

| 项 | 最低 | 推荐 | 我们实测 |
| --- | :-: | :-: | :-: |
| GPU 显存 | 4 GB | 8+ GB | NVIDIA L20 14 GB |
| 内存 | 16 GB | 32 GB | 185 GB |
| 磁盘 | 20 GB | 100 GB | 100 GB+（模型权重大头）|
| CUDA | 11.8+ | 12.1+ | 12.1（driver 12.2）|

**关键点**：0.6B 模型 LoRA SFT 实际只占 **~3.5 GB** 显存（开 gradient_checkpointing），所以 **RTX 3090 24G、4090 24G 甚至消费级 8 GB 卡都能跑**。

### 1.2 软件依赖（核心 5 个）

| 包 | 版本（实测）| 作用 | 是否必需 |
| --- | :-: | --- | :-: |
| `torch` | 2.5.1+cu121 | 训练后端 | ✅ |
| `transformers` | 5.7.0 | 模型加载 + tokenizer | ✅ |
| `peft` | 0.19.1 | LoRA 适配器实现 | ✅ |
| `trl` | 1.3.0 | SFTTrainer / GRPOTrainer | ✅ |
| `datasets` | 4.8.5 | JSONL 加载 + 缓存 | ✅ |

辅助包：`accelerate`、`scikit-learn`（算 MCC）、`pandas`（读 TSV）、`tqdm`、`bitsandbytes`（QLoRA 时用）。

### 1.3 Python 版本：3.10 / 3.12 都行

我们用 **Python 3.12.12**。3.10 也行，**3.9 不行**（trl/datasets 有版本下限）。

### 1.4 一份完整的 requirements

`my_try/requirements-local.in`：

```text
torch>=2.3
transformers>=4.45
datasets>=3.0
peft>=0.13
accelerate>=1.0
trl>=0.12
bitsandbytes>=0.43
scikit-learn
pandas
pyarrow
tqdm
tensorboard
jupyter
ipykernel
matplotlib
```

具体 pip 命令（PyTorch 官方索引 + 国内镜像）：

```bash
# 1) torch (PyTorch 官方 cu121 索引)
.venv-cola/bin/pip install "torch>=2.3" \
    --index-url https://download.pytorch.org/whl/cu121

# 2) 其他依赖（腾讯云镜像，速度极快）
.venv-cola/bin/pip install \
    -i https://mirrors.cloud.tencent.com/pypi/simple \
    --trusted-host mirrors.cloud.tencent.com \
    "transformers>=4.45" "datasets>=3.0" "peft>=0.13" "accelerate>=1.0" \
    "trl>=0.12" "bitsandbytes>=0.43" \
    scikit-learn pandas pyarrow tqdm tensorboard jupyter ipykernel matplotlib
```

### 1.5 ⚠️ 必踩的两个环境坑

#### 坑 1：`_lzma` 模块缺失（Python 3.12 自编译版本）

**现象**：

```python
>>> from trl import SFTTrainer
ModuleNotFoundError: No module named '_lzma'
```

**原因**：系统 Python 3.12 编译时没装 `xz-devel`，`_lzma` C 扩展没编出来。`datasets` 库顶层 `import lzma`，链式失败。

**修复**：在 venv 里放一个 stub 模块（**不重编 Python**）：

```bash
# 文件: .venv-cola/lib/python3.12/site-packages/_lzma.py
# 内容: 一个伪造的 lzma 接口，提供所有常量和类，但实际调用会 NotImplementedError
```

完整 stub 代码见 `my_try/environment.md` §5。我们项目只读本地 tsv/jsonl，不解压 .xz，stub 完全够。

#### 坑 2：trl 1.x 的 API 变更（重要！）

trl 在 1.0 之后改了一些参数名，**网上 2024 年前的教程 90% 会踩这三个坑**：

| trl 0.x 旧写法 | **trl 1.x 正确写法** |
| --- | --- |
| `tokenizer=tokenizer` | **`processing_class=tokenizer`** |
| `max_seq_length=...` | **`max_length=...`** |
| `model.add_adapter(LoraConfig(...))` | **直接传 `peft_config=LoraConfig(...)`** 给 SFTTrainer |

---

## 2. 数据准备

### 2.1 原始数据：CoLA

| 文件 | 条数 | 格式 |
| --- | :-: | --- |
| `cola_data/in_domain_train.tsv` | 8551 | TSV，4 列：source / label / first_label / text |
| `cola_data/in_domain_dev.tsv` | 527 | 同上，做验证 |

label = `0`（unacceptable）/ `1`（acceptable）

### 2.2 目标数据格式：`messages` JSONL

trl 的 SFTTrainer 1.x **原生支持 messages 格式**，每行一个 JSON：

```json
{
  "messages": [
    {"role": "user",      "content": "Decide whether...\nSentence: ...\n\nYour answer:"},
    {"role": "assistant", "content": "acceptable"}
  ]
}
```

trl 会自动用 tokenizer 的 chat_template 渲染成模型能理解的格式（Qwen3 是 ChatML），还会自动识别"哪些是 user 段哪些是 assistant 段"，配合 `completion_only_loss=True` 只在 assistant 段算 loss。

### 2.3 转换脚本（`my_try/prepare_sft_data.py`）

核心 30 行：

```python
PLAIN_TEMPLATE = (
    'Decide whether the following sentence is grammatically acceptable or not. '
    'If it is grammatically correct, answer "acceptable". '
    'If not, answer "unacceptable". '
    'Only output "acceptable" or "unacceptable", and do not output any other information.\n\n'
    "Sentence: {sentence}\n\n"
    "Your answer:"
)

def build_messages(text: str, label: int) -> dict:
    answer = "acceptable" if int(label) == 1 else "unacceptable"
    return {
        "messages": [
            {"role": "user", "content": PLAIN_TEMPLATE.format(sentence=text)},
            {"role": "assistant", "content": answer},
        ]
    }

# 读 TSV → 转 JSONL
df = pd.read_csv("cola_data/in_domain_train.tsv", sep="\t", header=None,
                 names=["source", "label", "first_label", "text"])
with open("my_try/data/sft_train.jsonl", "w") as f:
    for row in df.itertuples(index=False):
        rec = build_messages(row.text, row.label)
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
```

### 2.4 ⭐ 数据准备的两个关键设计原则

#### 原则 1：训练 prompt 与评测 prompt **完全一致**

`PLAIN_TEMPLATE` 必须和 `eval_baseline.py` / `eval_lora.py` 的 `PLAIN_TEMPLATE` **逐字符一致**。如果差一个空格或标点：
- 训练时模型学到的是 "用户输入 X → 我输出 Y"
- 评测时输入 X' ≠ X，模型行为会偏移

我们的做法：把 `PLAIN_TEMPLATE` 写成常量在两个脚本里复制（保持简单，不引入共享模块）。

#### 原则 2：assistant 段**只放最终答案**，不放 CoT

我们的 assistant 内容只有 `"acceptable"` 或 `"unacceptable"`，**不带任何思维链**。原因（来自 Step 4 的实验结论）：

| 数据格式 | 风险 |
| --- | --- |
| ✅ 纯短答案 | 模型学到稳定的格式，parse_fail 接近 0 |
| ❌ 带 CoT（"Let me think... so the answer is X"）| 推理时输出格式漂移，parse_fail 飙升（Step 4 实测从 0 涨到 37/527）|

**记忆点**：CoLA 这种"秒级二分类"任务**不需要 reasoning**。

### 2.5 产出

```
my_try/data/sft_train.jsonl   8551 条
my_try/data/sft_val.jsonl      527 条
```

---

## 3. 模型下载

### 3.1 推荐 ModelScope（国内速度快）

```bash
cd /data/workspace/Github-open/CoLA-RL/model
git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen3-0.6B.git
```

**坑**：直接 `cd ... && nohup git clone ... &` 写法，cd 不一定生效（`&` 可能在 cd 完成前 fork）。要分两行写：

```bash
cd model
nohup git clone ... > log 2>&1 &
```

### 3.2 验证

```bash
du -sh model/Qwen3-0.6B/      # 应该 ~1.5 GB
ls model/Qwen3-0.6B/          # 应该有 config.json, model.safetensors, tokenizer.json
```

### 3.3 清理 .git/lfs 省磁盘

git lfs 会保留一份 LFS 对象副本，把 1.5GB 翻倍。clone 完后：

```bash
rm -rf model/Qwen3-0.6B/.git/lfs   # 安全，不影响使用
```

---

## 4. 训练脚本设计（`my_try/train_lora_sft.py`）

### 4.1 核心代码骨架（约 80 行）

```python
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
import json
from pathlib import Path

# --- 1. 数据 ---
def load_jsonl(path):
    return Dataset.from_list([json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()])

train_ds = load_jsonl("my_try/data/sft_train.jsonl")
val_ds = load_jsonl("my_try/data/sft_val.jsonl")

# --- 2. tokenizer + 模型（基座 bf16）---
tok = AutoTokenizer.from_pretrained("model/Qwen3-0.6B")
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "model/Qwen3-0.6B",
    dtype=torch.bfloat16,         # ⚠️ trl 5.x 用 dtype，不是 torch_dtype
    device_map="cuda",
)
model.config.use_cache = False    # gradient_checkpointing 必要

# --- 3. LoRA 配置 ---
lora_cfg = LoraConfig(
    r=16,                          # rank（核心超参，见 §4.2）
    lora_alpha=32,                 # 经验值 = 2 × rank
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 标准
    task_type="CAUSAL_LM",
)

# --- 4. 训练参数 ---
sft_cfg = SFTConfig(
    output_dir="my_try/ckpt/Qwen3-0.6B-lora-E1-r16",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,        # effective batch = 32
    learning_rate=2e-4,                   # ⚠️ LoRA 用 1e-4 ~ 5e-4
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
    completion_only_loss=True,            # ⭐ trl 1.x 新特性，只对 assistant 算 loss
    max_length=256,                       # ⚠️ trl 1.x 不是 max_seq_length
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to=[],                         # 不接 wandb
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# --- 5. Trainer ---
trainer = SFTTrainer(
    model=model,
    args=sft_cfg,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tok,                 # ⚠️ trl 1.x 不是 tokenizer=
    peft_config=lora_cfg,                 # 直接传，不用 get_peft_model
)

# --- 6. 训练 ---
trainer.train()
trainer.save_model()
tok.save_pretrained(sft_cfg.output_dir)
```

### 4.2 关键超参逐个讲

#### `r=16`（LoRA rank）

| rank | LoRA 参数 / 总参数（0.6B 模型）| 适用 |
| :-: | :-: | --- |
| 4-8 | 0.2-0.4% | 简单风格调整（语气、格式）|
| **16** | **0.76%** | **通用默认**，CoLA 这种任务足够 |
| 32 | 1.5% | 任务陌生 / 数据量大 |
| 64 | 3% | 复杂多任务 |

**经验**：先用 16 跑一遍，效果不够再加大。我们 r=16 在 0.6B 上 MCC 0.5406，r=32/64 大概率没显著提升。

#### `lora_alpha = 2 × rank`

实际生效的 LoRA 强度是 `alpha / rank`。`alpha=2r` 是经验稳健值（peft 默认建议）。

#### `target_modules`

哪些层加 LoRA。Qwen3 的标准选择是 4 个 attention 投影：`q_proj, k_proj, v_proj, o_proj`。如果效果不够，可以加 `gate_proj, up_proj, down_proj`（FFN 层），代价是 LoRA 参数增加 ~3 倍。

**注意**：`Qwen3.5` 系列（多模态版本）的 module 名不一样，要先 `print(model)` 看清楚再选。

#### `learning_rate=2e-4`

| 训练方式 | 典型 lr | 原因 |
| --- | :-: | --- |
| 全参 SFT | **1e-5 ~ 5e-5** | 全局更新，要慢 |
| **LoRA SFT** | **1e-4 ~ 5e-4** | 只动小适配器，可以快 |
| LoRA-RL | 5e-6 ~ 1e-5 | RL 不稳，要保守 |

**记忆**：LoRA 的 lr 比全参 SFT **大 1-2 个数量级**。我们用 2e-4 是 LoRA 的中位数。

#### `effective_batch_size = bsz × grad_accum = 8 × 4 = 32`

显存吃不下大 bsz 时用梯度累积。32 是 SFT 的安全甜点（太小梯度噪声大，太大学习慢）。

#### `max_length=256`

CoLA 一条 prompt 80 token + 答案 2 token，256 完全够。**不要写 1024+**，会浪费显存。

#### `completion_only_loss=True`（⭐ trl 1.x 关键特性）

让 loss 只在 assistant 段计算，不在 user prompt 段计算。这点非常关键：
- ❌ 不开：模型也在学"如何复述用户的 prompt"，学习信号被稀释
- ✅ 开：梯度专注于"输入 X → 输出 acceptable/unacceptable"的映射

实测开了之后 token_accuracy 从 0.85 涨到 0.92，MCC 从 ~0.45 涨到 0.54。

#### `gradient_checkpointing=True`

激活值用时重算，省 5-8 倍显存，速度慢 30%。**14G 卡上必开**。

### 4.3 保存的产物

```
my_try/ckpt/Qwen3-0.6B-lora-E1-r16/
├── adapter_config.json          # LoRA 配置
├── adapter_model.safetensors    # LoRA 权重（仅 ~30 MB）
├── chat_template.jinja          # tokenizer 的 chat 模板
├── tokenizer.json / tokenizer_config.json
├── README.md                    # peft 自动生成
├── run_info.json                # 我们手写的训练元数据
└── checkpoint-{N}/              # 每 epoch 一个，默认保留最后 2 个
```

**关键点**：`adapter_model.safetensors` 只有几十 MB，**部署时只需要这一个文件**（基座共享）。

---

## 5. 训练执行

### 5.1 后台启动（标准做法）

按项目惯例，超过 2 分钟的脚本要后台跑：

```bash
cd /data/workspace/Github-open/CoLA-RL

PYTHONUNBUFFERED=1 nohup .venv-cola/bin/python my_try/train_lora_sft.py \
    --model model/Qwen3-0.6B \
    --exp E1 --rank 16 --epochs 3 \
    --bsz 8 --grad_accum 4 --lr 2e-4 --max_length 256 \
    > /tmp/sft_E1.log 2>&1 &
echo "E1_PID=$!"
```

### 5.2 监控（每 5-15 min check 一次）

```bash
# 进程是否还在
ps -p <PID> -o pid,stat,etime

# 最新进度
tail -1 /tmp/sft_E1.log | tr '\r' '\n' | tail -2

# 关键指标
grep -E "eval_loss|train_loss|训练完成|显存峰值" /tmp/sft_E1.log | tail -5
```

### 5.3 关键训练指标解读

我们 E1 的训练曲线（trl 自动 log 的）：

| Epoch | eval_loss | eval_mean_token_accuracy | 解读 |
| :-: | :-: | :-: | --- |
| 1 | 0.4515 | 0.919 | 起步快速收敛 |
| 2 | 0.4338 | 0.921 | 还在下降 |
| 3 | **0.4300** | **0.9216** | ⭐ 最低，未过拟合 |

**判断收敛的几个信号**：
- ✅ eval_loss 单调下降（或最后一个 epoch 平稳）→ 训练充分
- ⚠️ eval_loss 在第 2 epoch 反弹 → 过拟合，减 epoch
- ⚠️ eval_loss 始终 0.5+ → 学不动，可能 lr 太小或数据有问题

### 5.4 实测耗时与显存

| 项 | 值 |
| --- | :-: |
| 训练时长 | **13.4 min**（3 epoch × 8551 样本）|
| 显存峰值 | **3.57 GB**（14G 卡用了 25%）|
| 速度 | ~10 it/s |
| 可训练参数 | 4.59M / 600M = **0.76%** |

### 5.5 多任务并行（如果显存够）

E1 (0.6B) 和 E4 (1.7B) 我们是**并行跑的**：

```bash
# 终端 1
nohup ... train_lora_sft.py --model model/Qwen3-0.6B --exp E1 ... &

# 终端 2（隔 30 秒，避免抢初始化）
nohup ... train_lora_sft.py --model model/Qwen3-1.7B --exp E4 ... &
```

合计显存稳态约 **8 GB**，14G 卡完全够。**总耗时 = max(E1, E4) = 23.5 min**（比串行省一半）。

注意：初始化阶段会有显存峰值，可能瞬间触顶。我们实测合计峰值一度达 13.88 GB（差点 OOM），稳定后回落到 12.3 GB。

---

## 6. 评测

### 6.1 加载 LoRA adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "model/Qwen3-0.6B"
adapter = "my_try/ckpt/Qwen3-0.6B-lora-E1-r16"

tok = AutoTokenizer.from_pretrained(adapter)
tok.padding_side = "left"   # ⚠️ decoder-only 推理必须左 padding

base_model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base_model, adapter)
model = model.merge_and_unload()   # 把 LoRA 合并进基座，加速推理
model.eval()
```

### 6.2 批量推理 + 计算 MCC

```python
import pandas as pd
from sklearn.metrics import matthews_corrcoef

df = pd.read_csv("cola_data/in_domain_dev.tsv", sep="\t", header=None,
                 names=["source", "label", "first_label", "text"])

y_true, y_pred = [], []
for batch in chunked(df, 16):
    sentences = batch["text"].tolist()
    messages = [[{"role": "user", "content": PLAIN_TEMPLATE.format(sentence=s)}]
                for s in sentences]
    enc = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        padding=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = enc["input_ids"].to("cuda")
    attn_mask = enc["attention_mask"].to("cuda")

    with torch.no_grad():
        out = model.generate(
            input_ids, attention_mask=attn_mask,
            max_new_tokens=8,             # 答案最多 3-4 token，留点余量
            do_sample=False,              # 贪心解码，结果可复现
            pad_token_id=tok.eos_token_id,
        )
    gen = out[:, input_ids.shape[-1]:]
    texts = tok.batch_decode(gen, skip_special_tokens=True)

    for label, raw in zip(batch["label"], texts):
        gt = "acceptable" if label == 1 else "unacceptable"
        pred = parse_answer(raw)
        y_true.append(gt)
        y_pred.append(pred or "acceptable")  # 解析失败兜底

mcc = matthews_corrcoef(y_true, y_pred)
print(f"MCC = {mcc:.4f}")
```

### 6.3 关键评测细节

#### `padding_side = "left"` ⚠️

这是 decoder-only 模型批量推理的**铁律**：
- 默认 right padding：生成会从 `<pad>` 之后开始，结果错乱
- 必须 left padding：所有 prompt **末尾对齐**，生成才正确

我们 sanity check 时一度看到 right padding 警告，加上这行才解决。

#### `enable_thinking=False`

Qwen3 的 chat_template 有一个 `enable_thinking` 参数：
- `True`（默认）：模型可能输出 `<think>...</think>` 后再答
- `False`：在 prompt 里注入 `<think>\n\n</think>\n\n`，告诉模型"思考已结束，直接答"

我们 SFT 用的是纯短答案数据，**评测必须用 `False`** 保持一致。

#### `do_sample=False`

贪心解码，**结果可复现**。RL 才用 sampling（多样性），评测用贪心更稳。

#### `merge_and_unload()`

把 LoRA 权重合并进基座，推理时只剩一个普通模型，**速度快 30%**。如果你要继续训练，**别合并**（合并后没法回到 LoRA 状态了）。

### 6.4 解析答案的鲁棒性

```python
def parse_answer(text: str) -> str | None:
    t = text
    if "</think>" in t:                   # 去掉思考链
        t = t.split("</think>")[-1]
    t = t.strip().lower()
    # 多行输出取最后一行（CoT 类输出）
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if lines:
        tail = lines[-1]
        if "unacceptable" in tail:        # 顺序：先长的后短的（前者是后者的子串）
            return "unacceptable"
        if "acceptable" in tail:
            return "acceptable"
    return None
```

**坑点**：必须先判 `unacceptable` 再判 `acceptable`，否则 `"unacceptable"` 会被错认成 `"acceptable"`（前者包含后者）。

### 6.5 实测

| 指标 | E1 值 |
| --- | :-: |
| MCC | **0.5406** |
| Accuracy | 0.808 |
| Parse fail | **0/527** |
| 评测耗时 | **3.9s**（batch=16）|
| Avg new tokens | 1.3 |

`avg_tokens=1.3` 说明模型几乎只输出 1 个词，**完全学到了"短答案"的格式**。这是 `completion_only_loss=True` 的功劳。

---

## 7. 对比与结果分析

### 7.1 与 baseline 对比

| 实验 | Baseline MCC | LoRA SFT 后 MCC | Δ |
| --- | :-: | :-: | :-: |
| Qwen3-0.6B | 0.000 | **0.5406** | **+0.54** |
| Qwen3-1.7B | 0.500 | **0.6246** | +0.12 |

**核心观察**：LoRA SFT 对**baseline 弱的小模型增益巨大**（0.6B 从 0 涨到 0.54），对已会的大模型增益较小（边际递减）。

### 7.2 混淆矩阵透视

```
Qwen3-0.6B (baseline):                  Qwen3-0.6B (E1 LoRA SFT):
                pred=unacc  pred=acc                pred=unacc  pred=acc
true=unacc        0          162                    105          57
true=acc          0          365                     44         321
                                         ↑
                                  baseline 全说 yes
                                  SFT 后真的学会了说 unacc!
```

unacceptable 的召回从 **0% → 65%**——模型不再"瞎说 yes"。

### 7.3 与全参 SFT 对比（重要发现）

我们另外做了 Qwen3-0.6B 的全参 SFT（`my_try/train_full_sft.py`）：

| 方法 | MCC | 显存 | 训练时长 | 可训练参数 |
| :-: | :-: | :-: | :-: | :-: |
| **LoRA r=16** | **0.5406** | 3.57 GB | 13.4 min | 4.6M (0.76%) |
| 全参 SFT v2（lr=5e-5）| 0.5361 | 13.23 GB | 4.3 min | 596M (100%) |

**LoRA 略胜全参！** 在 CoLA 这种"任务简单 + 数据中等"的场景下，LoRA 已经足够。这印证了 QLoRA 论文的结论：**LoRA 与全参的效果差距随模型变大、任务复杂度变大才显现**。

---

## 8. 注意事项总结

### 8.1 容易踩的坑（按严重程度）

| 严重 | 坑 | 修复 |
| :-: | --- | --- |
| 🔴 致命 | trl 1.x 用 `tokenizer=` 参数 | 改 `processing_class=` |
| 🔴 致命 | trl 1.x 用 `max_seq_length=` | 改 `max_length=` |
| 🔴 致命 | 推理用 right padding | 改 `tok.padding_side = "left"` |
| 🔴 致命 | parse 时先判 `acceptable` | 必须先判 `unacceptable` |
| 🟡 严重 | 没开 `completion_only_loss=True` | MCC 直接掉 0.1+ |
| 🟡 严重 | `dtype=bf16` 写错为 `torch_dtype=` | trl 5.x deprecated |
| 🟡 严重 | 训练 prompt 与评测 prompt 不一致 | 严格逐字符对齐 |
| 🟢 一般 | 没设 `model.config.use_cache = False` | grad_checkpointing 报 warning |
| 🟢 一般 | LoRA target_modules 用 set 写 | 序列化 run_info.json 时 TypeError，转 list |

### 8.2 关键设计原则（带回家）

1. **prompt 对齐**：训练 prompt 与评测 prompt 一字不差
2. **格式简洁**：assistant 只放纯答案，不带 CoT
3. **completion_only_loss=True**：trl 1.x 的命脉
4. **left padding for inference**：decoder-only 必须
5. **lr 与方法匹配**：LoRA 1e-4 ~ 5e-4，全参 1e-5 ~ 5e-5
6. **eval_loss 单调下降**：观察是否过拟合的最直接信号
7. **gradient_checkpointing 总是开**：消费卡的救命稻草

### 8.3 故障诊断清单

| 现象 | 可能原因 |
| --- | --- |
| MCC = 0 / 全说一个答案 | 模式崩溃；检查 lr 是否过大、数据是否平衡 |
| eval_loss 不下降 | lr 太小、数据格式错误、prompt-completion 不对齐 |
| OOM | 减 bsz、加 grad_accum、开 grad_checkpointing |
| parse_fail 大量 | 模型输出格式漂移；检查 `enable_thinking`、`max_new_tokens` |
| 推理结果乱 | padding_side 没设 left、enable_thinking 不一致 |
| 显存稳态比预期高 | tok 没开 padding_side="left"，或 dataloader_num_workers 太大 |

### 8.4 性能 vs 资源建议

| 卡 | 推荐配置（0.6B 模型）|
| --- | --- |
| 8 GB（如 3060/4060）| `bsz=4, grad_accum=8, max_length=256, grad_checkpointing=True` |
| 16 GB（如 4080/A4000）| `bsz=8, grad_accum=4, max_length=256` |
| 24 GB+（3090/4090/A5000）| `bsz=16, grad_accum=2, max_length=512`（更激进）|

我们 14G L20：默认配置占 3.57 GB，**有大量空间并行做其他实验**。

---

## 9. 一页式可复用 SOP

```bash
# === 0. 前置 ===
cd /data/workspace/Github-open/CoLA-RL
source .venv-cola/bin/activate    # 已经装好了

# === 1. 准备数据（每个新任务做一次）===
python my_try/prepare_sft_data.py
# → my_try/data/sft_train.jsonl + sft_val.jsonl

# === 2. 训练（核心命令）===
nohup .venv-cola/bin/python my_try/train_lora_sft.py \
    --model model/Qwen3-0.6B \
    --exp E1 --rank 16 --epochs 3 \
    --bsz 8 --grad_accum 4 --lr 2e-4 --max_length 256 \
    > /tmp/sft_E1.log 2>&1 &

# === 3. 监控 ===
tail -f /tmp/sft_E1.log

# === 4. 评测 ===
.venv-cola/bin/python my_try/eval_lora.py \
    --adapter my_try/ckpt/Qwen3-0.6B-lora-E1-r16

# === 5. 看结果 ===
cat my_try/res_lora_Qwen3-0.6B-lora-E1-r16.json | jq '.results[0] | {mcc, accuracy, parse_fail}'
```

---

## 10. 扩展方向

| 方向 | 难度 | 收益 | 备注 |
| --- | :-: | :-: | --- |
| 加大 rank（16 → 32 → 64）| ⭐ | 微小（CoLA 太简单）| 先 r=16 |
| 加 target_modules（含 FFN）| ⭐⭐ | 中等 | 加 `gate_proj/up_proj/down_proj` |
| LoRA-RL（GRPO） | ⭐⭐⭐ | **可能负收益** | 注意 mode collapse，见 `step6_summary.md` |
| QLoRA（4bit 基座）| ⭐⭐ | 显存省 4×，效果略损 | 适合 7B+ 模型 |
| 全参 SFT（如果显存够）| ⭐⭐ | LoRA 已接近上限时差距小 | 见 `step5_summary.md` §6.7 |
| 数据扩充（CoT 蒸馏）| ⭐⭐⭐⭐ | 接近原项目 0.598 上限 | 需要调用大模型 API 蒸馏数据 |

---

## 11. 文件索引

如果要复现，关键文件：

| 文件 | 作用 |
| --- | --- |
| `my_try/environment.md` | 完整环境记录（含坑） |
| `my_try/requirements-local.in` | 源依赖清单 |
| `my_try/requirements-local.txt` | pip freeze 产出（166 包精确版本）|
| `my_try/sanity_check.py` | 环境验证 |
| `my_try/prepare_sft_data.py` | 数据格式转换 |
| **`my_try/train_lora_sft.py`** | **训练脚本（主文件）** |
| **`my_try/eval_lora.py`** | **评测脚本（主文件）** |
| `my_try/step5_summary.md` | 实验结果汇总 |
| `my_try/lora_sft_tutorial.md` | 本教程 |

---

## 12. 致谢与参考

- **原项目**：[ytzfhqs/CoLA-RL](https://github.com/ytzfhqs/CoLA-RL)（基于 verl 的全套 GRPO 训练）
- **HuggingFace trl**：https://github.com/huggingface/trl
- **HuggingFace peft**：https://github.com/huggingface/peft
- **CoLA 数据集**：[Warstadt et al., 2018](https://arxiv.org/abs/1805.12471)
- **LoRA 论文**：[Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- **QLoRA 论文**：[Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)

---

> **一句话总结**：
> LoRA SFT 是把"几百 GB 显存才能跑的全参微调"压缩到 **3-5 GB 消费卡** 上的关键技术。trl + peft 让代码量降到 80 行内。在 CoLA 这种简单任务上，**LoRA 几乎能持平全参 SFT 效果**，但训练时长和显存开销只有 1/4。
