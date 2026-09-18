#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/cuda_env.sh

bash scripts/31_verify_v2_1_listwise_freeze.sh

TRAIN_ROOT=results/v2_rank_then_cut/listwise_training
LOG_ROOT="$TRAIN_ROOT/logs"
mkdir -p "$LOG_ROOT"

launch() {
  local seed=$1
  local gpu=$2
  local name="train2000_seed${seed}"
  local output_dir="$TRAIN_ROOT/$name"
  local log_path="$LOG_ROOT/$name.log"

  if [[ -f "$output_dir/training_metadata.json" ]]; then
    echo "$name already complete"
    return
  fi
  if ps -ww -eo args | grep '[s]rc.training.train_atomic_listwise_ranker' \
      | grep -F -- "--output-dir $output_dir" >/dev/null; then
    echo "$name already running"
    return
  fi

  nohup setsid env CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 \
    .venv/bin/python -m src.training.train_atomic_listwise_ranker \
      --train-data results/v2_rank_then_cut/rank_learning_curve/train2000_rank_oracle.jsonl \
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

# Physical GPU4 is deliberately excluded. At the frozen launch snapshot GPU0
# had the only cool, low-utilization capacity; GPU3 had ample memory and lower
# utilization than GPUs 1, 2, and 5. Two small QLoRA jobs share GPU0 safely.
launch 20260910 0
launch 20260911 0
launch 20260912 3
