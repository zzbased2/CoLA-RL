# CoLA-RL 本地环境搭建记录

> 对应文档：`my_try.md` Step 2
> 搭建日期：2026-05-04
> 项目路径：`/data/workspace/Github-open/CoLA-RL`

---

## 1. 硬件 / 系统实况

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA L20 × 1 |
| 显存（total） | 15.03 GB（≈ 15360 MiB，比 `nvidia-smi` 报的 14336 MiB 多一点是 `cudaMemGetInfo` 与 spec 之差） |
| SM 数 | 92 |
| CUDA Driver | 535.161.07（对应 CUDA 12.2 runtime） |
| nvcc | CUDA 12.1.105 |
| OS | Linux / sh |
| 系统 Python | `/usr/bin/python3.9`（3.9.16，太旧） |
| 可用的新 Python | `/usr/local/python3.12/bin/python3.12`（3.12.12） |
| `/data` 磁盘 | 91 GB 可用（102 G 总量的 82% 已用） |

> 📌 `my_try.md` 原本提到复用 `browser-use-env/venv` 的 Python 3.12，实际上我们发现系统里有独立的 `/usr/local/python3.12`，直接基于它创建了项目专用 venv，更干净。

## 2. 虚拟环境

```bash
/usr/local/python3.12/bin/python3.12 -m venv /data/workspace/Github-open/CoLA-RL/.venv-cola

# 激活：
source /data/workspace/Github-open/CoLA-RL/.venv-cola/bin/activate
# 或直接用绝对路径：
/data/workspace/Github-open/CoLA-RL/.venv-cola/bin/python ...
```

venv 所在：`/data/workspace/Github-open/CoLA-RL/.venv-cola/`（约 **6.2 GB**，torch + CUDA 轮子是大头）

## 3. pip 镜像踩坑

- ❌ `pypi.tuna.tsinghua.edu.cn`（清华）：本机访问时 HTTP 403 / `No matching distribution found`，疑似 IP 段被限速或封禁
- ✅ `download.pytorch.org/whl/cu121`：可用，用于装 torch（带 CUDA 12.1 runtime 的 wheel）
- ✅ `mirrors.cloud.tencent.com/pypi/simple`：腾讯内网镜像，速度 **100+ MB/s**，强烈推荐

**最终安装命令**：

```bash
# 1) torch (从 PyTorch 官方 cu121 索引)
/data/workspace/Github-open/CoLA-RL/.venv-cola/bin/pip install "torch>=2.3" \
    --index-url https://download.pytorch.org/whl/cu121

# 2) 其他依赖（腾讯云镜像）
/data/workspace/Github-open/CoLA-RL/.venv-cola/bin/pip install \
    -i https://mirrors.cloud.tencent.com/pypi/simple \
    --trusted-host mirrors.cloud.tencent.com \
    "transformers>=4.45" "datasets>=3.0" "peft>=0.13" "accelerate>=1.0" \
    "trl>=0.12" "bitsandbytes>=0.43" \
    scikit-learn pandas pyarrow tqdm tensorboard jupyter ipykernel matplotlib
```

## 4. 实际装到的关键包版本（相对 `my_try.md §2.4`）

| 包 | 文档下限 | 实际装到 | 备注 |
| --- | --- | --- | --- |
| torch | >=2.3 | **2.5.1+cu121** | cu121 wheel |
| transformers | >=4.45 | **5.7.0** | ⚠️ 跳了大版本 |
| trl | >=0.12 | **1.3.0** | ⚠️ 跳了大版本 |
| peft | >=0.13 | **0.19.1** | |
| datasets | >=3.0 | **4.8.5** | |
| accelerate | >=1.0 | **1.13.0** | |
| bitsandbytes | >=0.43 | **0.49.2** | QLoRA 所需 |
| numpy | - | **2.4.4** | numpy 2.x，注意有 API 变动 |
| pandas | - | **3.0.2** | |

完整 166 个包的冻结清单：`requirements-local.txt`，源清单：`requirements-local.in`。

## 5. ⚠️ transformers 5.x / trl 1.x 的 API 变更（已踩，必须记）

**踩坑 1：`apply_chat_template(..., return_tensors="pt")` 返回值类型变更**

transformers 4.x 时代：

```python
inputs = tok.apply_chat_template(messages, return_tensors="pt", ...)  # 直接是 Tensor
inputs.shape[-1]            # ✅
inputs.to("cuda")           # ✅
model.generate(inputs, ...) # ✅
```

transformers 5.x：

```python
inputs = tok.apply_chat_template(messages, return_tensors="pt", ...)
# inputs 变成 BatchEncoding（dict-like），没有 .shape 属性
# model.generate(inputs, ...) 报 AttributeError: 'BatchEncoding' has no attribute 'shape'
```

**正确写法（在本项目所有脚本里统一用这个模板）**：

```python
enc = tok.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
    return_dict=True,           # ← 关键：强制 dict 返回
)
input_ids = enc["input_ids"].to("cuda")
attn_mask = enc.get("attention_mask")
if attn_mask is not None:
    attn_mask = attn_mask.to("cuda")

out = model.generate(
    input_ids,
    attention_mask=attn_mask,
    max_new_tokens=32,
    do_sample=False,
    pad_token_id=tok.eos_token_id,
)
resp = tok.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
```

**影响范围**：`my_try.md §3.3` 里提供的 `eval_baseline.py` 样例代码需要按上面模板改，否则 Step 3 一启动就 `AttributeError`。

**踩坑 2：deprecation 警告（不影响运行）**

```
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
```

以后所有 `AutoModelForXxx.from_pretrained(..., torch_dtype=...)` 建议改成 `dtype=...`，但 5.7.0 还兼容旧写法。

**踩坑 3（预警，Step 5 要注意）：`trl.SFTConfig` / `SFTTrainer` API**

trl 1.x 大版本里 `SFTConfig` 的部分字段名有调整（例如 `max_seq_length` → `max_length`，`tokenizer` 参数被 `processing_class` 替代等）。Step 5 写训练脚本时需要用 `python -c "from trl import SFTConfig; help(SFTConfig)"` 先确认当前签名。

## 6. Sanity check 结果

运行：

```bash
/data/workspace/Github-open/CoLA-RL/.venv-cola/bin/python /data/workspace/Github-open/CoLA-RL/my_try/sanity_check.py
```

输出（精简）：

```
torch 版本             : 2.5.1+cu121
CUDA 编译版本          : 12.1
CUDA 是否可用          : True
GPU                   : NVIDIA L20
显存（total）         : 15.03 GB
SM count              : 92

加载模型: /data/workspace/Github-open/CoLA-RL/model/Qwen3-0.6B
模型加载耗时           : 1.5s
模型权重显存占用       : 1.19 GB

模型回复               : 'Hello! How can I assist you today?'
当前已分配显存         : 1.20 GB
当前已 reserved 显存   : 1.67 GB

✅ sanity check 通过
```

结论：**14 G 卡跑 0.6B bf16 推理轻松**，推理阶段显存余量 13+ GB，Step 3 baseline 评测完全无压力。

## 7. 可复用的环境变量/激活命令

为方便后续所有步骤，建议在终端里设好：

```bash
export COLA_ROOT=/data/workspace/Github-open/CoLA-RL
export COLA_PY=$COLA_ROOT/.venv-cola/bin/python
export COLA_PIP=$COLA_ROOT/.venv-cola/bin/pip

# 运行脚本（从项目根目录执行）
cd $COLA_ROOT
$COLA_PY my_try/sanity_check.py
$COLA_PY my_try/eval_baseline.py --prompt plain
```

## 8. Step 2 完成状态

- [x] 创建 `.venv-cola` 虚拟环境（Python 3.12.12）
- [x] 安装 torch 2.5.1+cu121
- [x] 安装 transformers/trl/peft/accelerate/datasets/bitsandbytes 等
- [x] 生成 `requirements-local.in`（源清单）
- [x] 生成 `requirements-local.txt`（pip freeze 产出，166 包）
- [x] 编写并运行 `sanity_check.py` 验证通过
- [x] 记录本 `environment.md`

## 9. 下一步：Step 3 要注意的事

1. **务必按本文档 §5「踩坑 1」的模板写 `apply_chat_template`**，否则 `eval_baseline.py` 会挂
2. CoLA 验证集 `cola_data/in_domain_dev.tsv` 已就位（527 条）
3. 四个模型权重已就位：Qwen3-0.6B / Qwen3-1.7B / Qwen3.5-0.8B / Qwen3.5-2B
4. 非思考模式预期每模型 10-20 分钟，四个合计约 1 小时
