#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-Beauty}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2025}"
MAX_LEN="${MAX_LEN:-50}"
OUT_DIR="${OUT_DIR:-results/stage1_sanity}"

mkdir -p "$OUT_DIR"

python main.py \
  --model SASRec \
  --dataset "$DATASET" \
  --gpu_id "$GPU_ID" \
  --seed "$SEED" \
  --max_item_list_length "$MAX_LEN" \
  --checkpoint_dir "$OUT_DIR/ckpt/${DATASET}_L${MAX_LEN}_seed${SEED}" \
  2>&1 | tee "$OUT_DIR/${DATASET}_SASRec_L${MAX_LEN}_seed${SEED}_sanity.log"
