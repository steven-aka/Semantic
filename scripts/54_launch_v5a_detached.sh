#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
session=semantic_v5a_scheduler
if tmux has-session -t "$session" 2>/dev/null; then
  echo "V5-A scheduler already running in tmux session $session"
  exit 0
fi
mkdir -p results/v2_rank_then_cut/v5a_tail_cost_training
tmux new-session -d -s "$session" \
  "cd '$PWD' && bash scripts/53_run_v5a_tail_cost.sh 2>&1 | tee results/v2_rank_then_cut/v5a_scheduler.log"
echo "started V5-A scheduler in tmux session $session"

