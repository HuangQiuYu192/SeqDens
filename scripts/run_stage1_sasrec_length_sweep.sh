#!/usr/bin/env bash
set -euo pipefail

DATASETS="${DATASETS:-Beauty}"
LENGTHS="${LENGTHS:-50 100 200 500 1000}"
SEEDS="${SEEDS:-2025}"
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-300}"
STOPPING_STEP="${STOPPING_STEP:-10}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2048}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
N_LAYERS="${N_LAYERS:-2}"
N_HEADS="${N_HEADS:-2}"
INNER_SIZE="${INNER_SIZE:-256}"
LR="${LR:-0.001}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-results/stage1_sasrec_length_sweep/${RUN_TAG}}"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/ckpt"

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    for max_len in $LENGTHS; do
      run_name="${dataset}_SASRec_L${max_len}_seed${seed}"
      log_file="$OUT_DIR/logs/${run_name}.log"
      ckpt_dir="$OUT_DIR/ckpt/${run_name}"

      echo "===== Running ${run_name} ====="
      python main.py \
        --model SASRec \
        --dataset "$dataset" \
        --gpu_id "$GPU_ID" \
        --seed "$seed" \
        --max_item_list_length "$max_len" \
        --epochs "$EPOCHS" \
        --eval_step 1 \
        --stopping_step "$STOPPING_STEP" \
        --train_batch_size "$BATCH_SIZE" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --learning_rate "$LR" \
        --hidden_size "$HIDDEN_SIZE" \
        --n_layers "$N_LAYERS" \
        --n_heads "$N_HEADS" \
        --inner_size "$INNER_SIZE" \
        --hidden_dropout_prob 0.5 \
        --attn_dropout_prob 0.5 \
        --loss_type CE \
        --metrics Recall NDCG Hit MRR \
        --topk 10 \
        --valid_metric NDCG@10 \
        --eval_split LS \
        --eval_order TO \
        --eval_group_by user \
        --eval_valid_mode full \
        --eval_test_mode full \
        --show_progress false \
        --checkpoint_dir "$ckpt_dir" \
        2>&1 | tee "$log_file"
    done
  done
done

python scripts/collect_recbole_results.py "$OUT_DIR/logs" \
  --output "$OUT_DIR/stage1_sasrec_length_sweep_summary.csv"

echo "Summary written to $OUT_DIR/stage1_sasrec_length_sweep_summary.csv"
