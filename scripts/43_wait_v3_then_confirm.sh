#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/cuda_env.sh
root=results/v2_rank_then_cut
train="$root/v3_boundary_training"
log="$root/v3_rank_confirm_scheduler.log"

while true; do
  ready=1
  for seed in 20260910 20260911 20260912; do
    [[ -s "$train/train2000_seed${seed}/training_metadata.json" ]] || ready=0
  done
  (( ready == 1 )) && break
  sleep 60
done

.venv/bin/python - <<'PY'
import json
from pathlib import Path
from src.reproducibility import sha256, write_metadata
root=Path('results/v2_rank_then_cut/v3_boundary_training')
runs=[]
for seed in (20260910,20260911,20260912):
    path=root/f'train2000_seed{seed}'/'training_metadata.json'
    row=json.load(open(path))
    runs.append((float(row['best_validation_loss']),seed,path,row))
loss,seed,path,row=min(runs)
write_metadata('results/v2_rank_then_cut/v3_selected_ranker.json',{
    'selection_status':'selected before any rank-confirm model evaluation',
    'selection_rule':'minimum consumed-development validation worst-boundary loss across frozen seeds',
    'seed':seed,
    'best_epoch':row['best_epoch'],
    'best_validation_loss':loss,
    'checkpoint':str(path.parent),
    'training_metadata_sha256':sha256(path),
})
PY

read -r checkpoint seed < <(.venv/bin/python - <<'PY'
import json
x=json.load(open('results/v2_rank_then_cut/v3_selected_ranker.json'))
print(x['checkpoint'],x['seed'])
PY
)
gpu=$(nvidia-smi --query-gpu=index,memory.free,temperature.gpu \
  --format=csv,noheader,nounits \
  | awk -F', *' '$1 != 4 && $2 >= 12000 && $3 <= 89 {print $1, $2}' \
  | sort -k2,2nr | awk 'NR==1 {print $1}')
if [[ -z "$gpu" ]]; then
  echo "$(date '+%F %T %Z') no safe GPU for one-shot confirmation" >> "$log"
  exit 4
fi
echo "$(date '+%F %T %Z') selected seed${seed}; launching one-shot confirm on GPU${gpu}" >> "$log"
bash scripts/42_run_v3_rank_confirm.sh "$checkpoint" "$gpu" \
  > "$root/v3_rank_confirm_evaluation.log" 2>&1

.venv/bin/python -m src.evaluation.v3_boundary_decision \
  --summary "$root/v3_rank_confirm_evaluation/summary.json" \
  --checkpoint-metadata "$checkpoint/training_metadata.json" \
  --config configs/v3_critical_boundary_ranking.json \
  --output "$root/v3_rank_confirm_decision.json" \
  > "$root/v3_rank_confirm_decision.log" 2>&1
echo "$(date '+%F %T %Z') V3 one-shot confirmation decision complete" >> "$log"
