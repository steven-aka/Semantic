#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/v2_rank_then_cut/candidates5000_exact

exec 9>results/v2_rank_then_cut/candidates5000_exact/.shard4_wait.lock
if ! flock -n 9; then
  echo "another shard-4 waiter already owns the lock"
  exit 0
fi

active_shard4() {
  pgrep -af 'src.search.atomic_exact_search.*--num-shards 6.*--shard-index 4' >/dev/null
}

gpu_used_by_project() {
  local candidate_gpu="$1"
  local process_id visible
  while read -r process_id; do
    [[ -r "/proc/${process_id}/environ" ]] || continue
    visible="$(tr '\0' '\n' < "/proc/${process_id}/environ" | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -n 1)"
    [[ "$visible" == "$candidate_gpu" ]] && return 0
  done < <(pgrep -f 'src.search.atomic_exact_search' || true)
  return 1
}

best_free_gpu() {
  local candidate_gpu free_mib
  while read -r candidate_gpu free_mib; do
    gpu_used_by_project "$candidate_gpu" && continue
    if (( free_mib >= 32000 )); then
      echo "$candidate_gpu"
      return 0
    fi
  done < <(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | tr -d ' ' \
      | sort -t, -k2,2nr
  )
  return 1
}

echo "waiting for a project-unused GPU with >=32000 MiB free for V2 shard 4"
while true; do
  if active_shard4; then
    echo "shard 4 is already active; waiter exits"
    exit 0
  fi
  first_gpu="$(best_free_gpu || true)"
  if [[ -n "$first_gpu" ]]; then
    echo "candidate GPU ${first_gpu} found; requiring 30-second stable capacity"
    sleep 30
    second_gpu="$(best_free_gpu || true)"
    if [[ "$second_gpu" == "$first_gpu" ]] && ! active_shard4; then
      echo "launching shard 4 on physical GPU ${first_gpu} at $(date '+%F %T %Z')"
      source scripts/cuda_env.sh
      export CUDA_VISIBLE_DEVICES="$first_gpu"
      export HF_HUB_OFFLINE=1
      exec .venv/bin/python -m src.search.atomic_exact_search \
        --examples data/units/qampari_rank_v2_candidates5000.jsonl \
        --annotations data/units/qampari_rank_v2_candidates5000_annotations.jsonl \
        --packet-dir data/packets_qampari_rank_v2_candidates5000 \
        --output-dir results/v2_rank_then_cut/candidates5000_exact \
        --model models/Qwen3-8B \
        --batch-size 512 \
        --max-new-tokens 256 \
        --gpu-memory-utilization 0.45 \
        --max-model-len 4096 \
        --num-shards 6 \
        --shard-index 4
    fi
    echo "capacity changed during stability check; continuing to wait"
  else
    echo "[$(date '+%F %T %Z')] no safe GPU yet"
  fi
  sleep 30
done
