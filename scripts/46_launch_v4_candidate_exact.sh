#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/cuda_env.sh
bash scripts/45_verify_v4_relational_freeze.sh

root=results/v2_rank_then_cut
out="$root/v4_candidates500_exact"
logs="$root/v4_candidate_exact_logs"
mkdir -p "$out" "$logs"
gpus=(1 0)
for shard in 0 1; do
  gpu=${gpus[$shard]}
  if [[ "$gpu" == 4 ]]; then
    echo "physical GPU4 is excluded" >&2
    exit 2
  fi
  log="$logs/shard${shard}.log"
  pidfile="$logs/shard${shard}.pid"
  running=$(pgrep -f "[s]rc.search.atomic_exact_search.*v4_candidates500_exact.*--shard-index $shard --num-shards 2" | head -1 || true)
  if [[ -n "$running" ]]; then
    echo "shard $shard already running pid=$running"
    continue
  fi
  echo "[operator restart] $(date -Is) shard=$shard physical_gpu=$gpu" >>"$log"
  nohup setsid env CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    .venv/bin/python -m src.search.atomic_exact_search \
      --examples data/units/qampari_rank_v4_candidates500.jsonl \
      --annotations data/units/qampari_rank_v4_candidates500_annotations.jsonl \
      --packet-dir data/packets_qampari_rank_v4_candidates500 \
      --output-dir "$out" \
      --model models/Qwen3-8B \
      --batch-size 512 \
      --max-new-tokens 256 \
      --gpu-memory-utilization 0.45 \
      --max-model-len 4096 \
      --shard-index "$shard" \
      --num-shards 2 >>"$log" 2>&1 < /dev/null &
  echo "$!" >"$pidfile"
  echo "launched shard=$shard physical_gpu=$gpu pid=$!"
done
