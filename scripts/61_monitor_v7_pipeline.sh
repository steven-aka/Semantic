#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=results/v2_rank_then_cut
echo "V7 train3163 coverage test ($(date '+%F %T %Z'))"
cat "$root/v7_scheduler_state.json" 2>/dev/null || echo 'state: not started'
echo
for seed in 20260910 20260911 20260912; do
  log="$root/v7_4b_training/logs/train3163_seed${seed}.log"
  metadata="$root/v7_4b_training/train3163_seed${seed}/training_metadata.json"
  if [[ -s "$metadata" ]]; then
    .venv/bin/python - "$seed" "$metadata" <<'PY'
import json,sys
x=json.load(open(sys.argv[2]))
print(f"seed {sys.argv[1]}: complete, best_step={x['best_optimizer_step']}, best_validation_loss={x['best_validation_loss']:.6f}")
PY
  elif [[ -s "$log" ]]; then echo "seed $seed: running"; tail -n 3 "$log"
  else echo "seed $seed: pending"
  fi
done
echo
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
if [[ -s "$root/v7_selected_development_evaluation/summary.json" ]]; then echo; cat "$root/v7_selected_development_evaluation/summary.json"; fi
