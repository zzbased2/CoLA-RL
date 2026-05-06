#!/bin/bash
# Watchdog: 等 sweep_pid 完成，然后自动启动 Qwen3-4B QLoRA smoke test → 正式训练 → 评测
# 用法:  bash my_try/scripts/run_after_sweep.sh <sweep_pid>

set -u
cd /data/workspace/Github-open/CoLA-RL

SWEEP_PID=${1:-790450}
PY=.venv-cola/bin/python
LOGDIR=my_try/logs
RESDIR=my_try/results/step5_lora
mkdir -p $LOGDIR $RESDIR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "⏳ 等待 sweep PID=$SWEEP_PID 完成..."
while ps -p $SWEEP_PID > /dev/null 2>&1; do
    sleep 30
done
echo "✅ sweep 已完成，开始 Qwen3-4B QLoRA"

sleep 10  # 让 GPU 喘口气

# Step 1: smoke test，5 步看显存峰值
echo ""
echo "============================================================"
echo "🧪 Qwen3-4B QLoRA smoke test (5 steps)"
echo "============================================================"

$PY my_try/scripts/train_lora_sft.py \
    --model model/Qwen3-4B --exp E6-smoke --rank 16 \
    --load_in_4bit --optim adamw_8bit \
    --target_modules "q_proj,k_proj,v_proj,o_proj" \
    --bsz 2 --grad_accum 16 --epochs 1 \
    --lr 2e-4 --lr_scheduler cosine --warmup_ratio 0.03 \
    --max_length 256 --logging_steps 1 \
    --save_strategy no --max_train_samples 40 \
    > $LOGDIR/qwen4b_smoke.log 2>&1
RC=$?
echo ""
echo "smoke test exit code: $RC"
tail -20 $LOGDIR/qwen4b_smoke.log | tr '\r' '\n' | grep -E "loss|显存|GB|Error|OOM|完成" | tail -10

if [ $RC -ne 0 ]; then
    echo "❌ smoke test 失败，不继续正式训练"
    exit 1
fi

# 清理 smoke ckpt
rm -rf my_try/ckpt/Qwen3-4B-lora-E6-smoke-r16

# Step 2: 正式训练 (使用 Step1 sweep 选出的最佳配方)
# 默认先用"扩 MLP + r=32 + eff_bsz=64 + 3ep + lr=1e-4"，比较保守省显存
echo ""
echo "============================================================"
echo "🚀 Qwen3-4B QLoRA 正式训练 (E6)"
echo "============================================================"

$PY my_try/scripts/train_lora_sft.py \
    --model model/Qwen3-4B --exp E6 --rank 32 \
    --load_in_4bit --optim adamw_8bit \
    --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --bsz 2 --grad_accum 32 --epochs 3 \
    --lr 1e-4 --lr_scheduler cosine --warmup_ratio 0.03 \
    --max_length 256 --logging_steps 10 \
    --save_strategy epoch \
    > $LOGDIR/qwen4b_train.log 2>&1
RC=$?
if [ $RC -ne 0 ]; then
    echo "❌ 正式训练失败 (rc=$RC)"
    tail -30 $LOGDIR/qwen4b_train.log
    exit 1
fi
echo "✅ 训练完成"

# Step 3: 评测
echo ""
echo "============================================================"
echo "📊 Qwen3-4B QLoRA 评测"
echo "============================================================"

$PY my_try/scripts/eval_lora.py \
    --adapter my_try/ckpt/Qwen3-4B-lora-E6-r32 \
    --base model/Qwen3-4B \
    --batch_size 8 \
    --out $RESDIR/res_lora_Qwen3-4B-lora-E6-r32.json \
    > $LOGDIR/qwen4b_eval.log 2>&1
tail -10 $LOGDIR/qwen4b_eval.log

echo ""
echo "🎉 全部完成！结果: $RESDIR/res_lora_Qwen3-4B-lora-E6-r32.json"
