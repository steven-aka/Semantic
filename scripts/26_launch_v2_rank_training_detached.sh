#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/cuda_env.sh

TRAIN_ROOT=results/v2_rank_then_cut/training
LOG_ROOT="$TRAIN_ROOT/logs"
mkdir -p "$LOG_ROOT"

launch() {
  local size=$1
  local seed=$2
  local gpu=$3
  local name="train${size}_seed${seed}"
  local output_dir="$TRAIN_ROOT/$name"
  local log_path="$LOG_ROOT/$name.log"

  if [[ -f "$output_dir/training_metadata.json" ]]; then
    echo "$name already complete"
    return
  fi
  if ps -ww -eo args | grep '[s]rc.training.train_atomic_ranker' \
      | grep -F -- "--output-dir $output_dir" >/dev/null; then
    echo "$name already running"
    return
  fi

  nohup setsid env CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 \
    .venv/bin/python -m src.training.train_atomic_ranker \
      --train-data "results/v2_rank_then_cut/rank_learning_curve/train${size}_rank_oracle.jsonl" \
      --validation-data results/v2_rank_then_cut/ranking_validation300_rank_oracle.jsonl \
      --model models/Qwen3-1.7B \
      --output-dir "$output_dir" \
      --epochs 5 \
      --batch-size 1 \
      --gradient-accumulation 8 \
      --learning-rate 0.0001 \
      --max-length 4096 \
      --seed "$seed" \
      --head-hidden-size 256 \
      >"$log_path" 2>&1 </dev/null &
  echo "$name launched on physical GPU$gpu pid=$! log=$log_path"
}

# GPU4 is deliberately absent. GPU1 lacks safe free memory and GPU2 is under
# full external compute load at this launch snapshot.
launch 500 20260910 0
launch 500 20260911 3
launch 500 20260912 5
launch 1000 20260910 3
launch 1000 20260911 5
launch 1000 20260912 3
launch 2000 20260910 5
launch 2000 20260911 3
launch 2000 20260912 5
