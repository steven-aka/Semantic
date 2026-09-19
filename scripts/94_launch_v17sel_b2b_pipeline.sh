#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cuda_env.sh
root=results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout
mkdir -p "$root"
if [[ ! -s "$root/v8.pid" ]]; then
  echo "Expected the already-running lineage-clean V8 PID file" >&2
  exit 66
fi
if [[ -s "$root/pipeline.pid" ]] && kill -0 "$(cat "$root/pipeline.pid")" 2>/dev/null; then
  echo "Pipeline already running as PID $(cat "$root/pipeline.pid")"
  exit 0
fi
setsid nohup .venv/bin/python -u -m src.evaluation.v17sel_b2b_run_lineage \
  --gpu "${SEL_B2B_GPU:-0}" >"$root/pipeline.log" 2>&1 </dev/null &
pid=$!
echo "$pid" > "$root/pipeline.pid"
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  tail -25 "$root/pipeline.log" >&2
  exit 1
fi
echo "Automatic B2B pipeline started: PID $pid"
echo "Monitor: watch -n 10 bash scripts/93_monitor_v17sel_b2b.sh"
