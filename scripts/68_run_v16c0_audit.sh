#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/cuda_env.sh

out=results/v2_rank_then_cut/v16c0_deployed_fid_audit_v8_step250
mkdir -p "$out"
gpus=(${V16C0_GPUS:-0 1 2 5})
pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${gpus[$shard]} .venv/bin/python -m src.evaluation.v16c0_first_irreversible_divergence \
    --data results/v2_rank_then_cut/v13_internal_train2863_mask_value.jsonl \
    --exact-dir results/v2_rank_then_cut/candidates5000_exact \
    --checkpoint results/v2_rank_then_cut/v8_one_run_seed20260912/checkpoints/step250 \
    --output "$out/train2863_shard${shard}_details.jsonl" \
    --summary "$out/train2863_shard${shard}_summary.json" \
    --role train2863 --batch-size 8 --shard-index "$shard" --shard-count 4 \
    >"$out/train2863_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

.venv/bin/python -m src.evaluation.merge_v16c0_audit \
  --inputs "$out"/train2863_shard*_details.jsonl \
  --output "$out/train2863_details.jsonl" \
  --summary "$out/train2863_summary.json" \
  --role train2863
