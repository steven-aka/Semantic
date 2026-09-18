#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
bash scripts/47_monitor_v4_candidate_exact.sh
root=results/v2_rank_then_cut
tail_queue="$root/v4_candidate_exact_logs/gpu3_tail_examples.jsonl"
tail_log="$root/v4_candidate_exact_logs/gpu3_tail.log"
if [[ -s "$tail_queue" ]]; then
  tail_total=$(wc -l <"$tail_queue")
  tail_complete=$(python3 - "$tail_queue" "$root/v4_candidates500_exact" <<'PY'
import json
import sys
from pathlib import Path
queue = Path(sys.argv[1])
out = Path(sys.argv[2])
print(sum((out / f"{json.loads(line)['example_id']}.jsonl").is_file()
          for line in queue.read_text().splitlines() if line.strip()))
PY
)
  tail_last=$(grep -E '\[atomic exact ' "$tail_log" 2>/dev/null | tail -1 || true)
  if pgrep -f '[s]rc.search.atomic_exact_search.*gpu3_tail_examples.jsonl' >/dev/null; then
    echo "GPU3 tail worker: $tail_complete/$tail_total RUNNING ${tail_last:-before first completed example}"
  else
    echo "GPU3 tail worker: $tail_complete/$tail_total STOPPED_OR_COMPLETE ${tail_last:-no completed example}"
  fi
fi
if [[ -s "$root/v4_scheduler_state.json" ]]; then
  echo "Scheduler state:"
  cat "$root/v4_scheduler_state.json"
else
  echo "Scheduler state: NOT_STARTED"
fi
for seed in 20260910 20260911 20260912; do
  history="$root/v4_relational_training/train2000_seed${seed}/history.jsonl"
  log="$root/v4_relational_training/logs/train2000_seed${seed}.log"
  if [[ -s "$history" ]]; then
    echo "seed$seed: $(tail -1 "$history")"
  elif pgrep -f "[t]rain_relational_boundary_ranker.*--seed $seed" >/dev/null; then
    latest=$(grep -E '^\{"epoch"' "$log" 2>/dev/null | tail -1 || true)
    echo "seed$seed: RUNNING ${latest:-before first completed epoch}"
  else
    echo "seed$seed: PENDING"
  fi
done
