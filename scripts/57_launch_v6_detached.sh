#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
session=semantic_v6_scheduler
if tmux has-session -t "$session" 2>/dev/null; then
  echo "V6 scheduler already running in tmux session $session"
  exit 0
fi
tmux new-session -d -s "$session" \
  "cd '$PWD' && bash scripts/56_run_v6_4b_capacity.sh 2>&1 | tee results/v2_rank_then_cut/v6_scheduler.log"
echo "started V6 scheduler in tmux session $session"

