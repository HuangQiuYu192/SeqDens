#!/usr/bin/env bash
set -euo pipefail

DATASETS="${DATASETS:-Beauty}"
LENGTHS="${LENGTHS:-50 100 200 500 1000}"
SEEDS="${SEEDS:-2025}"
GPU_ID="${GPU_ID:-0}"
OUT_DIR="${OUT_DIR:-results/stage1_sasrec_length_sweep/$(date +%Y%m%d_%H%M%S)}"

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
        --checkpoint_dir "$ckpt_dir" \
        2>&1 | tee "$log_file"
    done
  done
done

python scripts/collect_recbole_results.py "$OUT_DIR/logs" \
  --output "$OUT_DIR/stage1_sasrec_length_sweep_summary.csv"

echo "Summary written to $OUT_DIR/stage1_sasrec_length_sweep_summary.csv"
