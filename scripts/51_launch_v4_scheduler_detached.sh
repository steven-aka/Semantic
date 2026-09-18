#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
root=results/v2_rank_then_cut
running=$(pgrep -f '[s]cripts/49_run_v4_after_exact.sh' | head -1 || true)
if [[ -n "$running" ]]; then
  echo "V4 scheduler already running pid=$running"
  exit 0
fi
session=semantic_v4_scheduler
if tmux has-session -t "$session" 2>/dev/null; then
  echo "V4 scheduler tmux session already exists: $session"
  exit 0
fi
log="$root/v4_scheduler.log"
tmux new-session -d -s "$session" -c "$PWD" \
  "bash scripts/49_run_v4_after_exact.sh > '$log' 2>&1"
echo "V4 scheduler launched in tmux session=$session log=$log"
