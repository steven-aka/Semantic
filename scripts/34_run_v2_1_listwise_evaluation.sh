#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/cuda_env.sh

seed=${1:?seed is required}
gpu=${2:?physical GPU index is required}
case "$seed" in
  20260910|20260911|20260912) ;;
  *) echo "unexpected seed: $seed" >&2; exit 2 ;;
esac
if [[ "$gpu" == 4 ]]; then
  echo "physical GPU4 is excluded" >&2
  exit 2
fi

checkpoint="results/v2_rank_then_cut/listwise_training/train2000_seed${seed}"
if [[ ! -s "$checkpoint/training_metadata.json" ]]; then
  echo "checkpoint is incomplete: $checkpoint" >&2
  exit 2
fi
mkdir -p results/v2_rank_then_cut/listwise_evaluation

CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 \
  .venv/bin/python -m src.evaluation.atomic_ranker_evaluation \
    --data results/v2_rank_then_cut/ranking_validation300_rank_oracle.jsonl \
    --exact-dir results/v2_rank_then_cut/candidates5000_exact \
    --checkpoint "$checkpoint" \
    --output "results/v2_rank_then_cut/listwise_evaluation/train2000_seed${seed}_details.jsonl" \
    --summary "results/v2_rank_then_cut/listwise_evaluation/train2000_seed${seed}_summary.json" \
    --model models/Qwen3-1.7B \
    --batch-size 1 \
    --max-length 4096 \
    --head-hidden-size 256
