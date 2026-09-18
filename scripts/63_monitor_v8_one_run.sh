#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
root=results/v2_rank_then_cut
echo "V8 one-run status"
date --iso-8601=seconds
[[ -s "$root/v8_one_run_state.json" ]] && cat "$root/v8_one_run_state.json"
if [[ -s "$root/v8_one_run.pid" ]]; then
  pid=$(cat "$root/v8_one_run.pid")
  if kill -0 "$pid" 2>/dev/null; then echo "process=RUNNING pid=$pid"; else echo "process=NOT_RUNNING pid=$pid"; fi
fi
echo
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
echo
[[ -s "$root/v8_one_run_seed20260912/history.jsonl" ]] && tail -3 "$root/v8_one_run_seed20260912/history.jsonl"
[[ -s "$root/v8_one_run_seed20260912/development_decision.json" ]] && cat "$root/v8_one_run_seed20260912/development_decision.json"
echo
[[ -s "$root/v8_one_run_seed20260912.log" ]] && tail -30 "$root/v8_one_run_seed20260912.log"
