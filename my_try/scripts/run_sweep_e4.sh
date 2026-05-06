#!/bin/bash
# Step 1: LoRA 超参扫描 (基座 Qwen3-1.7B)
# 串行跑 4 组 + 每组跑完立刻评测，失败一个不影响后续
# 日志写到 my_try/logs/sweep_e4_<exp>.log
# 结果写到 my_try/results/step5_lora/res_lora_Qwen3-1.7B-lora-<exp>-r<rank>.json

set -u
cd /data/workspace/Github-open/CoLA-RL

PY=.venv-cola/bin/python
MODEL=model/Qwen3-1.7B
LOGDIR=my_try/logs
RESDIR=my_try/results/step5_lora
mkdir -p $LOGDIR $RESDIR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ---------- 公共训练函数 ----------
train_and_eval() {
    local EXP=$1
    local RANK=$2
    local TARGETS=$3
    local BSZ=$4
    local ACCUM=$5
    local EPOCHS=$6
    local LR=$7
    local EXTRA="${8:-}"

    echo ""
    echo "============================================================"
    echo "🚀 $EXP : rank=$RANK targets=$TARGETS eff_bsz=$((BSZ*ACCUM)) epochs=$EPOCHS lr=$LR"
    echo "============================================================"

    # 训练
    $PY my_try/scripts/train_lora_sft.py \
        --model $MODEL --exp $EXP --rank $RANK \
        --target_modules "$TARGETS" \
        --bsz $BSZ --grad_accum $ACCUM \
        --epochs $EPOCHS --lr $LR \
        --lr_scheduler cosine --warmup_ratio 0.03 \
        --max_length 256 --logging_steps 10 \
        --save_strategy epoch \
        $EXTRA \
        > $LOGDIR/sweep_e4_${EXP}_train.log 2>&1
    local RC=$?
    if [ $RC -ne 0 ]; then
        echo "❌ $EXP 训练失败 (rc=$RC)，跳过评测"
        tail -20 $LOGDIR/sweep_e4_${EXP}_train.log
        return 1
    fi

    local CKPT=my_try/ckpt/Qwen3-1.7B-lora-${EXP}-r${RANK}
    echo "✅ $EXP 训练完成，开始评测..."

    # 评测
    $PY my_try/scripts/eval_lora.py \
        --adapter $CKPT \
        --base $MODEL \
        --out $RESDIR/res_lora_Qwen3-1.7B-lora-${EXP}-r${RANK}.json \
        > $LOGDIR/sweep_e4_${EXP}_eval.log 2>&1

    # 提取 MCC
    MCC=$($PY -c "
import json
with open('$RESDIR/res_lora_Qwen3-1.7B-lora-${EXP}-r${RANK}.json') as f:
    d=json.load(f)
r=d['results'][0] if 'results' in d else d
print(f\"MCC={r['mcc']:.4f}  Acc={r['accuracy']:.4f}\")
" 2>/dev/null)
    echo "📊 $EXP: $MCC"
}

# ---------- 4 组扫参 ----------

# E4-A：扩 target 到 MLP（gate/up/down）
train_and_eval "E4-A-mlp"     16 "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"  4 8  3 2e-4

# E4-B：大 batch 对齐原项目
train_and_eval "E4-B-bsz128"  16 "q_proj,k_proj,v_proj,o_proj"                              4 32 3 2e-4

# E4-C：高 rank
train_and_eval "E4-C-r32"     32 "q_proj,k_proj,v_proj,o_proj"                              4 8  3 2e-4

# E4-D：综合最优组合（扩 MLP + r=32 + bsz=128 + 5 ep + 降 lr）
train_and_eval "E4-D-best"    32 "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"  4 32 5 1e-4

echo ""
echo "============================================================"
echo "🎉 扫参完成，汇总结果："
echo "============================================================"
for F in $RESDIR/res_lora_Qwen3-1.7B-lora-E4-*-r*.json; do
    [ -f "$F" ] || continue
    $PY -c "
import json, os
with open('$F') as f:
    d=json.load(f)
r=d['results'][0] if 'results' in d else d
name=os.path.basename('$F').replace('res_lora_','').replace('.json','')
print(f\"{name:55s}  MCC={r['mcc']:.4f}  Acc={r['accuracy']:.4f}  parse_fail={r['parse_fail']}\")
"
done

# 对照组 baseline
echo "----"
echo "(baseline E4 r=16 qkvo eff=32 3ep lr=2e-4         MCC=0.6246  Acc=0.8387)"
