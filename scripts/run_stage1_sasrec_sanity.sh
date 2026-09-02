#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-Beauty}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2025}"
MAX_LEN="${MAX_LEN:-50}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2048}"
OUT_DIR="${OUT_DIR:-results/stage1_sanity}"

mkdir -p "$OUT_DIR"

python main.py \
  --model SASRec \
  --dataset "$DATASET" \
  --gpu_id "$GPU_ID" \
  --seed "$SEED" \
  --max_item_list_length "$MAX_LEN" \
  --epochs "$EPOCHS" \
  --eval_step 1 \
  --stopping_step 1 \
  --train_batch_size "$BATCH_SIZE" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --metrics Recall NDCG Hit MRR \
  --topk 10 \
  --valid_metric NDCG@10 \
  --eval_split LS \
  --eval_order TO \
  --eval_group_by user \
  --eval_valid_mode full \
  --eval_test_mode full \
  --show_progress false \
  --checkpoint_dir "$OUT_DIR/ckpt/${DATASET}_L${MAX_LEN}_seed${SEED}" \
  2>&1 | tee "$OUT_DIR/${DATASET}_SASRec_L${MAX_LEN}_seed${SEED}_sanity.log"
