"""
全参 SFT 训练脚本 (my_try.md §Step 5 - 补充对比实验)

对照 train_lora_sft.py 但不加 LoRA，直接更新所有参数。
目的：验证"LoRA 效果距离全参 SFT 有多远"。

在 14 GB L20 上跑 Qwen3-0.6B (0.6B × 11~12 倍 = ~13 GB 显存)，
属于"顶着显存上限"的实验，需要更保守的 batch / seq 配置。

用法:
  .venv-cola/bin/python my_try/train_full_sft.py \\
      --model model/Qwen3-0.6B --exp E1-FULL \\
      --bsz 4 --grad_accum 8 --epochs 3

关键与 LoRA 版的差异:
  - 无 peft_config
  - lr 从 LoRA 的 2e-4 降到全参 SFT 典型的 5e-6 - 2e-5（不这么降会训崩）
  - save_total_limit=1 节省磁盘（全参 ckpt 每份 ~1.2 GB）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_jsonl_as_dataset(path: Path) -> Dataset:
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
    p.add_argument("--model", required=True)
    p.add_argument("--exp", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bsz", type=int, default=4, help="per-device train batch size")
    p.add_argument("--grad_accum", type=int, default=8)
    # 全参 SFT 的学习率比 LoRA 小 1-2 个数量级
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler", default="cosine")
    p.add_argument("--train_file", default="my_try/data/sft_train.jsonl")
    p.add_argument("--val_file", default="my_try/data/sft_val.jsonl")
    p.add_argument("--output_root", default="my_try/ckpt")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_strategy", default="epoch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--optim", default="adamw_torch",
                   help="优化器；显存吃紧时用 adamw_8bit（bitsandbytes）"
                        "可把 1.7B 的 optimizer state 从 ~14GB 砍到 ~1.7GB")
    p.add_argument("--max_steps", type=int, default=-1,
                   help="smoke test 用，>0 则只跑这么多步")
    p.add_argument("--eval_strategy", default="epoch",
                   help="训练中 eval 策略；显存吃紧时设 'no' 关掉，可省 ~0.5-1GB 显存")
    args = p.parse_args()

    exp_name = f"{Path(args.model).name}-fullsft-{args.exp}"
    output_dir = Path(args.output_root) / exp_name

    print("=" * 60)
    print("🚀 全参 SFT 训练")
    print("=" * 60)
    print(f"  实验 ID        : {args.exp}")
    print(f"  基座模型       : {args.model}")
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
        print(f"⚠️  smoke-test: 只用前 {len(train_ds)} 条训练样本")
    print(f"\n📦 数据集: train={len(train_ds)}  val={len(val_ds)}")

    # --- 2. 模型 + tokenizer ---
    print(f"\n🧠 加载基座 {args.model} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.config.use_cache = False  # 开 gradient checkpointing 必要
    print(f"    加载耗时: {time.time() - t0:.1f}s；显存已占: "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # --- 3. SFTConfig ---
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
        completion_only_loss=True,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        save_total_limit=1,  # 全参 ckpt 太大，只留最新
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        # 全参 SFT 特有：可选 optimizer，默认 adamw_torch
        optim=args.optim,
        max_steps=args.max_steps,
    )

    # --- 4. Trainer（注意没有 peft_config）---
    trainer_kwargs = dict(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        processing_class=tok,
    )
    if args.eval_strategy != "no":
        trainer_kwargs["eval_dataset"] = val_ds
    trainer = SFTTrainer(**trainer_kwargs)

    trainable, total = 0, 0
    for _, p_ in trainer.model.named_parameters():
        total += p_.numel()
        if p_.requires_grad:
            trainable += p_.numel()
    print(f"\n🔧 可训练参数: {trainable:,} / {total:,} = {trainable/total*100:.2f}% (全参)")

    # --- 5. 训练 ---
    print(f"\n🚂 开始训练（{args.epochs} epoch）...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\n✅ 训练完成，耗时 {elapsed:.1f}s = {elapsed/60:.1f} 分钟")

    # --- 6. 保存 ---
    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))
    print(f"📦 全参模型已保存到: {output_dir}")

    info = {
        "exp": args.exp,
        "mode": "full_sft",
        "base_model": args.model,
        "epochs": args.epochs,
        "effective_bsz": args.bsz * args.grad_accum,
        "lr": args.lr,
        "lr_scheduler": args.lr_scheduler,
        "warmup_ratio": args.warmup_ratio,
        "optim": args.optim,
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

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"📊 显存峰值: {peak_gb:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
