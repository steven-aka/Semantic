#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
session=semantic_v7_scheduler
if tmux has-session -t "$session" 2>/dev/null; then
  echo "$session already exists"
  exit 0
fi
tmux new-session -d -s "$session" "cd '$PWD' && bash scripts/59_run_v7_data_scale.sh 2>&1 | tee results/v2_rank_then_cut/v7_scheduler.log"
echo "launched $session"
