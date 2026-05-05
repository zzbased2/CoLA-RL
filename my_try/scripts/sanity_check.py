"""
CoLA-RL 环境 sanity check 脚本
用于验证：
1. PyTorch + CUDA 可用
2. 显存总量
3. transformers 能正常加载 Qwen3-0.6B（bf16）并给出权重占用
4. 随便 forward 一次，确认 GPU 上能跑推理
"""
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.environ.get(
    "SANITY_MODEL",
    "/data/workspace/Github-open/CoLA-RL/model/Qwen3-0.6B",
)


def main() -> None:
    print("=" * 60)
    print("PyTorch sanity check")
    print("=" * 60)
    print(f"torch 版本             : {torch.__version__}")
    print(f"CUDA 编译版本          : {torch.version.cuda}")
    print(f"CUDA 是否可用          : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("❌ CUDA 不可用，后续步骤没意义，终止。")

    dev_idx = 0
    props = torch.cuda.get_device_properties(dev_idx)
    total_gb = props.total_memory / 1e9
    print(f"GPU                   : {props.name}")
    print(f"显存（total）         : {total_gb:.2f} GB")
    print(f"SM count              : {props.multi_processor_count}")

    print()
    print("=" * 60)
    print(f"加载模型: {MODEL_PATH}")
    print("=" * 60)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    mdl.eval()
    load_sec = time.time() - t0
    mem_gb = mdl.get_memory_footprint() / 1e9
    print(f"模型加载耗时           : {load_sec:.1f}s")
    print(f"模型权重显存占用       : {mem_gb:.2f} GB")

    print()
    print("=" * 60)
    print("跑一次 generate 验证前向")
    print("=" * 60)
    messages = [{"role": "user", "content": "Say hi in one short English sentence."}]
    # transformers 5.x: apply_chat_template(return_tensors="pt") 返回 BatchEncoding
    # 统一改为 return_dict=True，再手动取 input_ids 张量
    try:
        enc = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
    except TypeError:
        enc = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    input_ids = enc["input_ids"].to("cuda")
    attn_mask = enc.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to("cuda")

    with torch.no_grad():
        out = mdl.generate(
            input_ids,
            attention_mask=attn_mask,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    resp = tok.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"模型回复               : {resp!r}")

    alloc_gb = torch.cuda.memory_allocated(dev_idx) / 1e9
    reserved_gb = torch.cuda.memory_reserved(dev_idx) / 1e9
    print(f"当前已分配显存         : {alloc_gb:.2f} GB")
    print(f"当前已 reserved 显存   : {reserved_gb:.2f} GB")

    print()
    print("✅ sanity check 通过")


if __name__ == "__main__":
    main()