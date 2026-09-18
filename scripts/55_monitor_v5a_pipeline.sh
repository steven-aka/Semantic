#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
root=results/v2_rank_then_cut
echo "V5-A tail-risk + cost-aware ranking ($(date '+%F %T %Z'))"
if [[ -f "$root/v5a_scheduler_state.json" ]]; then
  cat "$root/v5a_scheduler_state.json"
else
  echo "state: not started"
fi
echo
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits
echo
for seed in 20260910 20260911 20260912; do
  log="$root/v5a_tail_cost_training/logs/train2000_seed${seed}.log"
  metadata="$root/v5a_tail_cost_training/train2000_seed${seed}/training_metadata.json"
  if [[ -s "$metadata" ]]; then
    .venv/bin/python - "$seed" "$metadata" <<'PY'
import json,sys
x=json.load(open(sys.argv[2]))
print(f"seed {sys.argv[1]}: complete, best_epoch={x['best_epoch']}, best_validation_loss={x['best_validation_loss']:.6f}")
PY
  elif [[ -s "$log" ]]; then
    echo "seed $seed: running"
    tail -n 2 "$log"
  else
    echo "seed $seed: pending"
  fi
done
if [[ -f "$root/v5a_selected_development_evaluation/summary.json" ]]; then
  echo
  cat "$root/v5a_selected_development_evaluation/summary.json"
fi
