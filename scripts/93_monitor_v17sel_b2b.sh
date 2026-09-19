#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
root=results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout
if [[ -s "$root/pipeline_status.json" ]]; then
  echo "Automatic lineage pipeline:"
  cat "$root/pipeline_status.json"
  echo
fi
if [[ -s "$root/pipeline.pid" ]]; then
  pipeline_pid="$(cat "$root/pipeline.pid")"
  if ps -p "$pipeline_pid" >/dev/null 2>&1; then
    ps -p "$pipeline_pid" -o pid,etimes,stat,cmd
  else
    echo "Pipeline PID $pipeline_pid is not running"
  fi
fi
if [[ -s "$root/pipeline.log" ]]; then
  echo "Latest pipeline log lines:"
  tail -5 "$root/pipeline.log"
fi
if [[ -s "$root/pipeline_status.json" ]]; then
  stage="$(.venv/bin/python -c 'import json; print(json.load(open("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/pipeline_status.json"))["stage"])')"
  if [[ -s "$root/$stage.log" ]]; then
    echo "Latest $stage log lines:"
    tail -5 "$root/$stage.log"
  fi
fi
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
  echo "V8 training stage complete"
fi
