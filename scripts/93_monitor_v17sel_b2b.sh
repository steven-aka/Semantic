#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
root=results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout
pid_file="$root/v8.pid"
log="$root/v8_run.log"
history="$root/v8_run/history.jsonl"

if [[ -s "$pid_file" ]]; then
  pid="$(cat "$pid_file")"
  if ps -p "$pid" >/dev/null 2>&1; then
    ps -p "$pid" -o pid,etimes,%cpu,%mem,stat,cmd
  else
    echo "V8 PID $pid is not running"
  fi
else
  echo "V8 PID file absent"
fi

if [[ -s "$history" ]]; then
  echo "Latest V8 checkpoint summary:"
  tail -1 "$history"
else
  echo "V8 has not reached its first 250-step validation checkpoint"
fi

if [[ -s "$log" ]]; then
  echo "Latest V8 log lines:"
  tail -8 "$log"
fi

if [[ -s "$root/v8_run/selected_checkpoint.json" ]]; then
  echo "V8 stage complete; V10/V12/V13 lineage stages have not yet been launched"
fi
