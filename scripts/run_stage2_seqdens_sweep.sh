#!/usr/bin/env bash
set -euo pipefail

DATASETS="${DATASETS:-ML-1M}"
MAX_LENS="${MAX_LENS:-500 1000}"
COMPRESSION_LENGTHS="${COMPRESSION_LENGTHS:-64}"
RECENT_LENGTHS="${RECENT_LENGTHS:-32}"
SEEDS="${SEEDS:-2025}"
GPU_ID="${GPU_ID:-0}"
OUT_DIR="${OUT_DIR:-results/stage2_seqdens_sweep/$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/ckpt"

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    for max_len in $MAX_LENS; do
      for compression_len in $COMPRESSION_LENGTHS; do
        for recent_len in $RECENT_LENGTHS; do
          run_name="${dataset}_SeqDensSASRec_L${max_len}_H${compression_len}_R${recent_len}_seed${seed}"
          log_file="$OUT_DIR/logs/${run_name}.log"
          ckpt_dir="$OUT_DIR/ckpt/${run_name}"

          echo "===== Running ${run_name} ====="
          python main.py \
            --model SeqDensSASRec \
            --dataset "$dataset" \
            --gpu_id "$GPU_ID" \
            --seed "$seed" \
            --max_item_list_length "$max_len" \
            --compression_length "$compression_len" \
            --recent_length "$recent_len" \
            --checkpoint_dir "$ckpt_dir" \
            2>&1 | tee "$log_file"
        done
      done
    done
  done
done

python scripts/collect_recbole_results.py "$OUT_DIR/logs" \
  --output "$OUT_DIR/stage2_seqdens_sweep_summary.csv"

echo "Summary written to $OUT_DIR/stage2_seqdens_sweep_summary.csv"
