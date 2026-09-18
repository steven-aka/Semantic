#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/cuda_env.sh
bash scripts/39_verify_v3_boundary_freeze.sh

root=results/v2_rank_then_cut/v3_boundary_training
mkdir -p "$root/logs"

launch() {
  local seed=$1
  local gpu=$2
  local output="$root/train2000_seed${seed}"
  local log="$root/logs/train2000_seed${seed}.log"
  if [[ "$gpu" == 4 ]]; then
    echo "physical GPU4 is excluded" >&2
    exit 2
  fi
  if [[ -s "$output/training_metadata.json" ]]; then
    echo "seed${seed} already complete"
    return
  fi
  if ps -ww -eo args | grep '[s]rc.training.train_critical_boundary_ranker' \
      | grep -F -- "--output-dir $output" >/dev/null; then
    echo "seed${seed} already running"
    return
  fi
  nohup setsid env CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 \
    .venv/bin/python -m src.training.train_critical_boundary_ranker \
      --train-data results/v2_rank_then_cut/v3_train2000_boundary_oracle.jsonl \
      --validation-data results/v2_rank_then_cut/v3_development300_boundary_oracle.jsonl \
      --model models/Qwen3-1.7B \
      --output-dir "$output" \
      --epochs 5 \
      --batch-size 1 \
      --gradient-accumulation 8 \
      --learning-rate 0.0001 \
      --max-length 4096 \
      --seed "$seed" \
      --head-hidden-size 256 \
      >"$log" 2>&1 </dev/null &
  echo "seed${seed} launched on physical GPU${gpu} pid=$!"
}

# Freeze the launch mapping at execution time after checking physical GPU
# memory. Two QLoRA jobs may safely share a completely empty A6000.
mapfile -t gpu_rows < <(
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits \
  | awk -F', *' '$1 != 4 && $2 >= 12000 && $4 <= 89 {print $1, $2, $3}' \
  | sort -k2,2nr
)
if (( ${#gpu_rows[@]} < 2 )); then
  echo "fewer than two safe non-GPU4 devices are available" >&2
  exit 3
fi
read -r gpu_a free_a _ <<<"${gpu_rows[0]}"
read -r gpu_b _ <<<"${gpu_rows[1]}"
gpu_c=$gpu_a
if (( free_a < 30000 )) && (( ${#gpu_rows[@]} >= 3 )); then
  read -r gpu_c _ <<<"${gpu_rows[2]}"
fi

launch 20260910 "$gpu_a"
launch 20260911 "$gpu_b"
launch 20260912 "$gpu_c"

