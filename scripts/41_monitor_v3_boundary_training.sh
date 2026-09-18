#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
root=results/v2_rank_then_cut/v3_boundary_training
complete=0
running=0
pending=0

echo "V3 critical-boundary ranking ($(date '+%F %T %Z'))"
for seed in 20260910 20260911 20260912; do
  output="$root/train2000_seed${seed}"
  log="$root/logs/train2000_seed${seed}.log"
  if [[ -s "$output/training_metadata.json" ]]; then
    value=$(.venv/bin/python -c \
      'import json,sys; x=json.load(open(sys.argv[1])); print("best_epoch={} validation_boundary_loss={:.6f} elapsed={:.2f}h".format(x["best_epoch"],x["best_validation_loss"],x["elapsed_seconds"]/3600))' \
      "$output/training_metadata.json")
    echo "seed${seed}: COMPLETE $value"
    complete=$((complete + 1))
  elif pid=$(ps -ww -eo pid=,args= | grep '[s]rc.training.train_critical_boundary_ranker' \
      | grep -F -- "--output-dir $output" | awk 'NR==1 {print $1}'); [[ -n "$pid" ]]; then
    epoch=$(tr '\r' '\n' < "$log" 2>/dev/null | grep '^{"epoch"' | tail -n 1 || true)
    if [[ -n "$epoch" ]]; then
      value=$(printf '%s' "$epoch" | .venv/bin/python -c \
        'import json,sys; x=json.load(sys.stdin); print("epoch={}/5 val_boundary_loss={:.6f} val_boundary_success={:.4f}".format(x["epoch"],x["validation_example_mean_worst_boundary_loss"],x["validation_complete_boundary_separation_fraction"]))')
    else
      value=initializing_or_epoch1
    fi
    elapsed=$(ps -p "$pid" -o etime= | xargs)
    echo "seed${seed}: RUNNING pid=$pid elapsed=$elapsed $value"
    running=$((running + 1))
  else
    echo "seed${seed}: PENDING_OR_STOPPED"
    pending=$((pending + 1))
  fi
done
echo "summary: complete=$complete/3 running=$running pending_or_stopped=$pending"
echo
if [[ -s results/v2_rank_then_cut/v3_selected_ranker.json ]]; then
  .venv/bin/python - <<'PY'
import json
x=json.load(open('results/v2_rank_then_cut/v3_selected_ranker.json'))
print('selected checkpoint: seed{} validation_boundary_loss={:.6f}'.format(x['seed'],x['best_validation_loss']))
PY
else
  echo "selected checkpoint: WAITING_FOR_ALL_TRAINING"
fi
if [[ -s results/v2_rank_then_cut/v3_rank_confirm_decision.json ]]; then
  .venv/bin/python - <<'PY'
import json
x=json.load(open('results/v2_rank_then_cut/v3_rank_confirm_decision.json'))
print('one-shot rank-confirm decision:',x['decision'])
PY
elif ps -ww -eo args | grep '[s]rc.evaluation.atomic_ranker_evaluation' \
    | grep -F 'v3_rank_confirm_evaluation' >/dev/null; then
  echo "one-shot rank-confirm: RUNNING"
else
  echo "one-shot rank-confirm: NOT_STARTED"
fi
echo
echo "GPU index, used MiB, free MiB, utilization %, temperature C"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits
echo "Physical GPU4 is excluded from this project by operator instruction."

