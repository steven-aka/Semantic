#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/cuda_env.sh

checkpoint=${1:?checkpoint is required}
gpu=${2:?physical GPU index is required}
if [[ "$gpu" == 4 ]]; then
  echo "physical GPU4 is excluded" >&2
  exit 2
fi
if [[ ! -s "$checkpoint/training_metadata.json" ]]; then
  echo "checkpoint is incomplete: $checkpoint" >&2
  exit 2
fi
out=results/v2_rank_then_cut/v3_rank_confirm_evaluation
mkdir -p "$out"
if [[ -e "$out/summary.json" || -e "$out/details.jsonl" ]]; then
  echo "one-shot V3 rank-confirm output already exists" >&2
  exit 3
fi

CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 \
  .venv/bin/python -m src.evaluation.atomic_ranker_evaluation \
    --data results/v2_rank_then_cut/rank_confirm210_rank_oracle.jsonl \
    --exact-dir results/v2_rank_then_cut/candidates5000_exact \
    --checkpoint "$checkpoint" \
    --output "$out/details.jsonl" \
    --summary "$out/summary.json" \
    --model models/Qwen3-1.7B \
    --batch-size 1 \
    --max-length 4096 \
    --head-hidden-size 256

