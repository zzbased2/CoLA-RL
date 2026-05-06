"""
Step 6: LoRA-GRPO 训练脚本（在 E1 LoRA SFT 基础上继续做 RL 对齐）

框架：trl 1.3.0 GRPOTrainer（纯 PyTorch generate，不用 vLLM）
基座：Qwen3-0.6B
起点：my_try/ckpt/Qwen3-0.6B-lora-E1-r16/ （SFT 后 MCC=0.5406）

核心差异 vs train_lora_sft.py：
- 换 SFTTrainer → GRPOTrainer
- 传 reward_funcs（Python callable）而非 labeled data
- prompts-only 数据集（没有 assistant 答案）
- 关键参数：num_generations=4（而非默认 8，省显存）
- beta=0.0（trl 1.x 默认 Dr.GRPO 风格，不用 Ref 模型，再省一份权重！）
- use_vllm=False（纯 PyTorch，避免独占显存）

显存预估（Qwen3-0.6B, num_gen=4, max_completion=128, bsz=1）:
  - Actor 权重 + LoRA + Adam      : ~2 GB
  - Ref 模型                       : 0 (beta=0)
  - Rollout (PyTorch generate)     : ~2 GB
  - 激活 (4 samples × 128 tokens)  : ~2-3 GB
  - 杂项                           : ~1 GB
  合计 ~7-8 GB，14 GB 卡充裕

用法（从项目根）:
  .venv-cola/bin/python my_try/train_lora_grpo.py \\
      --base_model model/Qwen3-0.6B \\
      --sft_adapter my_try/ckpt/Qwen3-0.6B-lora-E1-r16 \\
      --exp E1-GRPO --epochs 1 --num_generations 4 --max_completion_length 32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


# --- Prompt 模板（与 eval_baseline.py / prepare_sft_data.py 完全一致）---
PLAIN_TEMPLATE = (
    'Decide whether the following sentence is grammatically acceptable or not. '
    'If it is grammatically correct, answer "acceptable". '
    'If not, answer "unacceptable". '
    'Only output "acceptable" or "unacceptable", and do not output any other information.\n\n'
    "Sentence: {sentence}\n\n"
    "Your answer:"
)

# v5 专用：强制 CoT prompt，鼓励模型先推理再给答案
COT_TEMPLATE = (
    'Decide whether the following sentence is grammatically acceptable or not. '
    'First, analyze the sentence step by step (subject-verb agreement, argument structure, '
    'word order, idiomaticity, etc.). '
    'Then, on the LAST line, output ONLY ONE word: "acceptable" or "unacceptable".\n\n'
    "Sentence: {sentence}\n\n"
    "Analysis:"
)


def parse_answer(text: str) -> str | None:
    """与 eval_baseline.py 完全一致的解析逻辑"""
    t = text
    if "</think>" in t:
        t = t.split("</think>")[-1]
    t = t.strip().lower()
    non_empty_lines = [ln for ln in t.splitlines() if ln.strip()]
    if non_empty_lines:
        tail = non_empty_lines[-1]
        if "unacceptable" in tail:
            return "unacceptable"
        if "acceptable" in tail:
            return "acceptable"
    if "unacceptable" in t:
        return "unacceptable"
    if "acceptable" in t:
        return "acceptable"
    return None


def make_reward_fn(balanced: bool = False):
    """构造奖励函数。

    trl GRPOTrainer 期望 reward_funcs 的签名是:
      fn(completions: list[str], **kwargs) -> list[float]

    kwargs 会包含 dataset 中除 "prompt" 外的所有字段，
    我们的 dataset 里存了 "ground_truth"，trl 会自动把它作为 kwarg 传进来。

    两种 reward:
      - balanced=False  (v1, 原项目 cola.py 风格):
          答对 +1，答错/解析失败 -1
          ⚠️ 在不平衡数据上会 mode collapse 到"全输出多数类"

      - balanced=True   (v2, class-balanced):
          acceptable 答对  → +1.0
          unacceptable 答对 → +2.378  (= 0.704/0.296，逆频率加权)
          答错或解析失败    → -1.0
          → "全 yes" 期望 = 0.704*1 + 0.296*(-1) = +0.408
            "全 no" 期望 = 0.704*(-1) + 0.296*2.378 = +0.000
            "完美预测" 期望 = 0.704*1 + 0.296*2.378 = +1.408
          → 完美预测的回报是"全 yes"的 3.5×，破除 mode collapse 陷阱
    """
    # CoLA 训练集 8551 条中 acceptable=6023 (70.4%) / unacceptable=2528 (29.6%)
    WEIGHT_ACC = 1.0
    WEIGHT_UNACC = 0.704 / 0.296  # ≈ 2.378

    if balanced:
        def reward_fn(completions, ground_truth, **kwargs):
            rewards = []
            for comp, gt in zip(completions, ground_truth):
                pred = parse_answer(comp)
                if pred is None:
                    rewards.append(-1.0)
                elif pred == gt:
                    rewards.append(WEIGHT_ACC if gt == "acceptable" else WEIGHT_UNACC)
                else:
                    rewards.append(-1.0)
            return rewards
    else:
        def reward_fn(completions, ground_truth, **kwargs):
            rewards = []
            for comp, gt in zip(completions, ground_truth):
                pred = parse_answer(comp)
                if pred is None:
                    rewards.append(-1.0)
                elif pred == gt:
                    rewards.append(1.0)
                else:
                    rewards.append(-1.0)
            return rewards

    return reward_fn


def load_cola_as_prompts(tsv_path: str, tokenizer, max_samples: int = 0,
                         enable_thinking: bool = False,
                         prompt_template: str = "plain") -> Dataset:
    """把 CoLA TSV 加载为 GRPO 需要的 prompts-only 数据集。

    prompt_template:
      - "plain": 原 prompt，要求短答案（v1-v4 用）
      - "cot":   要求 step-by-step 分析后给答案（v5 用）

    enable_thinking=True 时，Qwen3 chat_template **不会**主动注入 `<think>\\n\\n</think>\\n\\n`
    占位符，让模型自己生成完整推理链 + 答案。这是原项目 verl 用的方式。
    """
    import pandas as pd
    template = COT_TEMPLATE if prompt_template == "cot" else PLAIN_TEMPLATE
    df = pd.read_csv(
        tsv_path, sep="\t", header=None,
        names=["source", "label", "first_label", "text"],
        dtype={"source": str, "label": int, "first_label": str, "text": str},
        keep_default_na=False,
    )
    if max_samples > 0:
        df = df.head(max_samples)

    rows = []
    for row in df.itertuples(index=False):
        user_content = template.format(sentence=row.text)
        messages = [{"role": "user", "content": user_content}]
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        gt = "acceptable" if int(row.label) == 1 else "unacceptable"
        rows.append({"prompt": prompt_str, "ground_truth": gt, "sentence": row.text})
    return Dataset.from_list(rows)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base_model", required=True,
                   help="基座路径，如 model/Qwen3-0.6B")
    p.add_argument("--sft_adapter", default=None,
                   help="SFT 阶段训练的 LoRA adapter 目录（E1）；"
                        "传入则作为 GRPO 起点，不传则从基座开始")
    p.add_argument("--exp", required=True, help="实验 ID，如 E1-GRPO")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj")

    p.add_argument("--epochs", type=int, default=1,
                   help="GRPO 通常只跑 1-3 epoch")
    p.add_argument("--bsz", type=int, default=1,
                   help="per_device_train_batch_size（prompt 数，每个会采样 num_gen 次）")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--num_generations", type=int, default=4,
                   help="每个 prompt 采样几次（对应原项目 rollout.n）")
    p.add_argument("--max_completion_length", type=int, default=32,
                   help="生成回复最大长度。CoLA 答案 1-2 token，32 够用")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="GRPO lr 比 LoRA-SFT 小一些，避免崩")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.0,
                   help="KL 系数。默认 0 = Dr.GRPO 风格（无 Ref 模型最省显存）")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)

    p.add_argument("--train_tsv", default="cola_data/in_domain_train.tsv")
    p.add_argument("--max_train_samples", type=int, default=0,
                   help=">0 时只取前 N 条；smoke test 用小值")
    p.add_argument("--output_root", default="my_try/ckpt")
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--reward_balanced",
        action="store_true",
        help="启用 class-balanced reward (acc=+1, unacc=+2.378, 错=-1)，"
             "破除不平衡分类的 mode collapse 陷阱（v2）",
    )
    p.add_argument(
        "--enable_thinking",
        action="store_true",
        help="启用 Qwen3 thinking 模式（chat_template 不注入 </think>，让模型自己生成 "
             "<think>...</think>{答案}），需配合 max_completion_length=1024+ 使用。"
             "这是原项目 verl 用的方式（v5）。",
    )
    p.add_argument(
        "--prompt_template",
        choices=["plain", "cot"],
        default="plain",
        help="prompt 模板：plain（短答案，v1-v4 用）或 cot（要求 step-by-step 分析，v5 用）",
    )
    args = p.parse_args()

    alpha = args.alpha if args.alpha is not None else 2 * args.rank
    exp_name = f"{Path(args.base_model).name}-grpo-{args.exp}-r{args.rank}"
    output_dir = Path(args.output_root) / exp_name

    print("=" * 60)
    print("🚀 LoRA-GRPO 训练（trl 1.3 + 纯 PyTorch rollout）")
    print("=" * 60)
    print(f"  实验 ID              : {args.exp}")
    print(f"  基座                 : {args.base_model}")
    print(f"  SFT adapter (起点)   : {args.sft_adapter or '(无，从基座开始)'}")
    print(f"  LoRA rank / alpha    : {args.rank} / {alpha}")
    print(f"  epochs               : {args.epochs}")
    print(f"  bsz × acc × num_gen  : {args.bsz} × {args.grad_accum} × {args.num_generations}")
    print(f"  max_completion_length: {args.max_completion_length}")
    print(f"  lr                   : {args.lr}")
    print(f"  beta (KL)            : {args.beta}  ({'无 Ref' if args.beta == 0 else '有 Ref'})")
    print(f"  输出目录             : {output_dir}")
    print("=" * 60)

    # --- 1. Tokenizer ---
    tok_src = args.sft_adapter or args.base_model
    print(f"\n📝 tokenizer 来源: {tok_src}")
    tok = AutoTokenizer.from_pretrained(tok_src)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only 生成必备

    # --- 2. 数据集 ---
    print(f"\n📦 加载训练 prompts: {args.train_tsv}")
    train_ds = load_cola_as_prompts(args.train_tsv, tok, args.max_train_samples,
                                    enable_thinking=args.enable_thinking,
                                    prompt_template=args.prompt_template)
    print(f"   样本数: {len(train_ds)}")
    print(f"   首条 ground_truth: {train_ds[0]['ground_truth']}")
    print(f"   首条 prompt (200 chars): {train_ds[0]['prompt'][:200]}")

    # --- 3. 模型（基座 + SFT adapter，再叠 GRPO LoRA）---
    print(f"\n🧠 加载基座 {args.base_model} ...")
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="cuda"
    )
    base_model.config.use_cache = False

    if args.sft_adapter:
        print(f"🔧 叠加 SFT adapter: {args.sft_adapter}")
        # 先加载 SFT adapter 并 merge 到基座（成为新的"基座"），然后 GRPO 会在上面再训一层 LoRA
        sft_model = PeftModel.from_pretrained(base_model, args.sft_adapter)
        model = sft_model.merge_and_unload()
        print(f"   SFT adapter 已合并到基座权重")
    else:
        model = base_model

    print(f"   模型加载耗时: {time.time() - t0:.1f}s；显存: "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # --- 4. LoRA config for GRPO ---
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=alpha,
        lora_dropout=args.dropout,
        bias="none",
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        task_type="CAUSAL_LM",
    )

    # --- 5. GRPOConfig ---
    grpo_cfg = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        # GRPO 专属
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        use_vllm=False,
        # 通用
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        log_completions=True,
        num_completions_to_print=2,
    )

    # --- 6. Trainer ---
    reward_fn = make_reward_fn(balanced=args.reward_balanced)
    print(f"\n🏆 构造 reward function (balanced={args.reward_balanced})")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_cfg,
        train_dataset=train_ds,
        processing_class=tok,
        peft_config=lora_cfg,
    )

    trainable, total = 0, 0
    for _, pp in trainer.model.named_parameters():
        total += pp.numel()
        if pp.requires_grad:
            trainable += pp.numel()
    print(f"🔧 可训练参数: {trainable:,} / {total:,} = {trainable/total*100:.3f}%")

    # --- 7. 训练 ---
    print(f"\n🚂 开始 GRPO 训练...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\n✅ 训练完成，耗时 {elapsed:.1f}s = {elapsed/60:.1f} 分钟")

    # --- 8. 保存 ---
    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))
    print(f"📦 LoRA-GRPO adapter 已保存: {output_dir}")

    info = {
        "exp": args.exp,
        "mode": "lora_grpo",
        "base_model": args.base_model,
        "sft_adapter": args.sft_adapter,
        "lora_rank": args.rank,
        "lora_alpha": alpha,
        "target_modules": sorted(list(lora_cfg.target_modules)),
        "epochs": args.epochs,
        "effective_bsz": args.bsz * args.grad_accum,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "lr": args.lr,
        "beta": args.beta,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "reward_balanced": args.reward_balanced,
        "enable_thinking": args.enable_thinking,
        "prompt_template": args.prompt_template,
        "train_samples": len(train_ds),
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
