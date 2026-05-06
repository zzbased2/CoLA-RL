"""
LoRA SFT 训练脚本 (my_try.md §Step 5)

- 基于 trl 1.3.0 SFTTrainer + peft LoraConfig
- 从 my_try/data/sft_train.jsonl 加载 messages 格式数据
- 输出到 my_try/ckpt/<name>/

用法示例 (从项目根目录执行)：

  # E1: Qwen3-0.6B rank=16
  .venv-cola/bin/python my_try/train_lora_sft.py \\
      --model model/Qwen3-0.6B --exp E1 --rank 16

  # E4: Qwen3-1.7B rank=16
  .venv-cola/bin/python my_try/train_lora_sft.py \\
      --model model/Qwen3-1.7B --exp E4 --rank 16

关键设计：
  - bf16 训练，基座冻结，只训 LoRA (target_modules = q/k/v/o_proj)
  - gradient_checkpointing=True 省激活
  - completion_only_loss=True  ← 只对 assistant 回复算 loss (trl 1.x 新特性)
    避免学习用户 prompt 的"回放"，让梯度专注于"acceptable / unacceptable" 的映射
  - max_length=256 (prompt ~80 token + 1-2 token 答案，绰绰有余)
  - per_device_train_batch_size=8, grad_accum=4 → 有效 bsz=32
  - epochs=3 (my_try.md §5.5 的 baseline 配置)

与 environment.md §5 "踩坑 3" 对应的 trl 1.x API:
  - tokenizer → processing_class
  - max_seq_length → max_length
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_jsonl_as_dataset(path: Path) -> Dataset:
    """加载 messages 格式 jsonl → datasets.Dataset"""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="基座模型路径，如 model/Qwen3-0.6B")
    p.add_argument("--exp", required=True, help="实验 ID，用于 ckpt 目录命名，如 E1")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--alpha", type=int, default=None, help="LoRA alpha，默认 = 2 × rank")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bsz", type=int, default=8, help="per-device train batch size")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler", default="cosine")
    p.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="LoRA 注入的模块，逗号分隔",
    )
    p.add_argument(
        "--train_file", default="my_try/data/sft_train.jsonl"
    )
    p.add_argument(
        "--val_file", default="my_try/data/sft_val.jsonl"
    )
    p.add_argument(
        "--output_root", default="my_try/ckpt", help="checkpoint 根目录"
    )
    p.add_argument(
        "--logging_steps", type=int, default=10
    )
    p.add_argument(
        "--eval_steps", type=int, default=0,
        help="每 N 步做一次验证；0 表示只在每 epoch 结束时评估"
    )
    p.add_argument(
        "--save_strategy", default="epoch", choices=["epoch", "steps", "no"]
    )
    p.add_argument(
        "--seed", type=int, default=42
    )
    p.add_argument(
        "--max_train_samples", type=int, default=0,
        help=">0 时只取前 N 条训练样本（smoke test 用）"
    )
    p.add_argument(
        "--load_in_4bit", action="store_true",
        help="启用 QLoRA：base model 以 nf4 量化加载，省显存（4B 模型必开）"
    )
    p.add_argument(
        "--optim", default="adamw_torch",
        help="优化器；QLoRA 建议 adamw_8bit / paged_adamw_8bit"
    )
    args = p.parse_args()

    alpha = args.alpha if args.alpha is not None else 2 * args.rank
    exp_name = f"{Path(args.model).name}-lora-{args.exp}-r{args.rank}"
    output_dir = Path(args.output_root) / exp_name

    print("=" * 60)
    print("🚀 LoRA SFT 训练")
    print("=" * 60)
    print(f"  实验 ID        : {args.exp}")
    print(f"  基座模型       : {args.model}")
    print(f"  LoRA rank      : {args.rank}  alpha={alpha}  dropout={args.dropout}")
    print(f"  target_modules : {args.target_modules}")
    print(f"  训练样本       : {args.train_file}")
    print(f"  验证样本       : {args.val_file}")
    print(f"  epochs         : {args.epochs}")
    print(f"  bsz (dev × acc): {args.bsz} × {args.grad_accum} = {args.bsz * args.grad_accum}")
    print(f"  max_length     : {args.max_length}")
    print(f"  lr             : {args.lr}  scheduler={args.lr_scheduler}")
    print(f"  输出目录       : {output_dir}")
    print("=" * 60)

    # --- 1. 数据 ---
    train_ds = load_jsonl_as_dataset(Path(args.train_file))
    val_ds = load_jsonl_as_dataset(Path(args.val_file))
    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
        print(f"⚠️  smoke-test 模式：只使用前 {len(train_ds)} 条训练样本")
    print(f"\n📦 数据集: train={len(train_ds)}  val={len(val_ds)}")
    print(f"    首条样例: {train_ds[0]['messages'][-1]}")

    # --- 2. 模型 + tokenizer ---
    print(f"\n🧠 加载基座 {args.model} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token  # Qwen 等模型没有 pad_token

    model_kwargs = dict(
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_cfg
        print(f"    🔢 启用 QLoRA：nf4 + double_quant + bf16 compute")

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False  # 开 gradient checkpointing 的必要条件
    if args.load_in_4bit:
        # QLoRA 必要步骤：prepare_model_for_kbit_training 开启 input_require_grads
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    print(f"    加载耗时: {time.time() - t0:.1f}s；显存已占: "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # --- 3. LoRA config (直接传给 SFTTrainer 的 peft_config) ---
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=alpha,
        lora_dropout=args.dropout,
        bias="none",
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        task_type="CAUSAL_LM",
    )

    # --- 4. Trainer args ---
    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        bf16=True,
        gradient_checkpointing=True,
        # 关键：让 trl 自动识别 messages 格式，并只对 assistant 回复算 loss
        completion_only_loss=True,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy="epoch" if args.eval_steps == 0 else "steps",
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        save_total_limit=2,
        report_to=[],  # 本次不接 wandb/tensorboard
        seed=args.seed,
        dataloader_num_workers=2,
        # 显存优化
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        optim=args.optim,
    )

    # --- 5. Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,       # trl 1.x: 取代旧的 tokenizer= 参数
        peft_config=lora_cfg,       # 由 trl 内部调 get_peft_model
    )

    # 打印可训练参数占比
    trainable, total = 0, 0
    for _, p_ in trainer.model.named_parameters():
        total += p_.numel()
        if p_.requires_grad:
            trainable += p_.numel()
    print(f"\n🔧 可训练参数: {trainable:,} / {total:,} = {trainable/total*100:.3f}%")

    # --- 6. 训练 ---
    print(f"\n🚂 开始训练（预计按 {args.epochs} epoch）...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\n✅ 训练完成，耗时 {elapsed:.1f}s = {elapsed/60:.1f} 分钟")

    # --- 7. 保存 ---
    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))
    print(f"📦 LoRA adapter 已保存到: {output_dir}")

    # 保存一份简要 run_info.json 便于后续追踪
    info = {
        "exp": args.exp,
        "base_model": args.model,
        "lora_rank": args.rank,
        "lora_alpha": alpha,
        "target_modules": sorted(list(lora_cfg.target_modules)),
        "epochs": args.epochs,
        "effective_bsz": args.bsz * args.grad_accum,
        "lr": args.lr,
        "max_length": args.max_length,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": trainable / total,
        "train_elapsed_sec": round(elapsed, 1),
        "output_dir": str(output_dir),
    }
    with (output_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"📝 run_info.json 已写入")

    # 显存峰值
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"📊 显存峰值: {peak_gb:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
