#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/cuda_env.sh

root=results/v2_rank_then_cut
mkdir -p "$root/listwise_evaluation/logs"

project_busy_gpus() {
  local pid value
  for pid in $(pgrep -f 'src.training.train_atomic_listwise_ranker|src.evaluation.atomic_ranker_evaluation' || true); do
    [[ -r "/proc/$pid/environ" ]] || continue
    value=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -n 1)
    [[ "$value" =~ ^[0-9]+$ ]] && echo "$value"
  done | sort -u
}

select_gpu() {
  local busy
  busy=$(project_busy_gpus | tr '\n' ' ')
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu,temperature.gpu \
      --format=csv,noheader,nounits \
    | awk -F',' -v busy=" $busy " '
        {
          for (i=1; i<=NF; i++) gsub(/^ +| +$/, "", $i)
          if ($1 == 4 || $2 < 10000 || $4 > 89) next
          if (index(busy, " " $1 " ")) next
          print $3, -$2, $1
        }' \
    | sort -n -k1,1 -k2,2 \
    | awk 'NR==1 {print $3}'
}

while true; do
  complete=0
  for seed in 20260910 20260911 20260912; do
    checkpoint="$root/listwise_training/train2000_seed${seed}/training_metadata.json"
    summary="$root/listwise_evaluation/train2000_seed${seed}_summary.json"
    log="$root/listwise_evaluation/logs/train2000_seed${seed}.log"
    if [[ -s "$summary" ]]; then
      complete=$((complete + 1))
      continue
    fi
    if [[ ! -s "$checkpoint" ]]; then
      continue
    fi
    if ps -ww -eo args | grep '[s]rc.evaluation.atomic_ranker_evaluation' \
        | grep -F "listwise_evaluation/train2000_seed${seed}" >/dev/null; then
      continue
    fi
    if [[ -s "$log" ]] && grep -Eiq 'Traceback|out of memory|CUDA error' "$log"; then
      echo "fatal prior evaluation failure for seed $seed; inspect $log" >&2
      exit 1
    fi
    gpu=$(select_gpu || true)
    if [[ -z "$gpu" ]]; then
      continue
    fi
    nohup setsid bash scripts/34_run_v2_1_listwise_evaluation.sh "$seed" "$gpu" \
      >"$log" 2>&1 </dev/null &
    echo "$(date '+%F %T %Z') evaluation seed$seed launched on physical GPU$gpu pid=$!"
    sleep 2
  done
  if [[ "$complete" == 3 ]]; then
    .venv/bin/python -m src.evaluation.listwise_ranking_decision \
      --run "$root/listwise_evaluation/train2000_seed20260910_summary.json" "$root/listwise_evaluation/train2000_seed20260910_details.jsonl" "$root/listwise_training/train2000_seed20260910/training_metadata.json" \
      --run "$root/listwise_evaluation/train2000_seed20260911_summary.json" "$root/listwise_evaluation/train2000_seed20260911_details.jsonl" "$root/listwise_training/train2000_seed20260911/training_metadata.json" \
      --run "$root/listwise_evaluation/train2000_seed20260912_summary.json" "$root/listwise_evaluation/train2000_seed20260912_details.jsonl" "$root/listwise_training/train2000_seed20260912/training_metadata.json" \
      --output "$root/v2_1_listwise_decision.json" \
      > "$root/v2_1_listwise_decision.log" 2>&1
    echo "$(date '+%F %T %Z') V2.1 listwise decision complete"
    exit 0
  fi
  sleep 60
done
