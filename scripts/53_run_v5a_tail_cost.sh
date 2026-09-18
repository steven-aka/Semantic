#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/cuda_env.sh
root=results/v2_rank_then_cut
config=configs/v5a_tail_risk_cost_aware_ranking.json
state="$root/v5a_scheduler_state.json"

write_state() {
  .venv/bin/python - "$state" "$1" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
path=Path(sys.argv[1])
value={"status":sys.argv[2],"updated_at":datetime.now().astimezone().isoformat()}
tmp=path.with_suffix(path.suffix+'.tmp')
tmp.write_text(json.dumps(value,indent=2)+'\n')
tmp.replace(path)
PY
}

write_state VERIFYING_FROZEN_V5A
.venv/bin/python - "$config" <<'PY'
import hashlib,json,sys
from pathlib import Path
config=json.loads(Path(sys.argv[1]).read_text())
for path,expected in config['implementation_sha256'].items():
    actual=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'frozen implementation changed: {path} {actual} != {expected}')
for role in ('train','development'):
    path=config['data'][role]
    actual=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    expected=config['data'][f'{role}_sha256']
    if actual != expected:
        raise SystemExit(f'frozen data changed: {path} {actual} != {expected}')
print('V5-A frozen implementation and data hashes: PASS')
PY

available_training_gpus() {
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
    --format=csv,noheader,nounits \
    | awk -F', *' '$1 != 4 && $2 >= 30000 && $3 <= 20 {print $1, $2}' \
    | sort -k2,2nr | awk '{print $1}'
}

train_seed() {
  local seed=$1 gpu=$2
  local out="$root/v5a_tail_cost_training/train2000_seed${seed}"
  mkdir -p "$root/v5a_tail_cost_training/logs"
  if [[ -s "$out/training_metadata.json" ]]; then
    echo "reuse complete V5-A seed $seed"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 \
    .venv/bin/python -m src.training.train_tail_cost_relational_ranker \
      --train-data "$root/v5_train2000_tail_cost_oracle.jsonl" \
      --validation-data "$root/v5_development300_tail_cost_oracle.jsonl" \
      --model models/Qwen3-1.7B \
      --output-dir "$out" \
      --epochs 5 --batch-size 4 --gradient-accumulation 2 \
      --learning-rate 0.0001 --max-length 4096 \
      --seed "$seed" --head-hidden-size 256 \
      --tail-fraction 0.25 --rate-lambda 0.25 \
      >"$root/v5a_tail_cost_training/logs/train2000_seed${seed}.log" 2>&1
}

write_state WAITING_FOR_THREE_TRAIN_GPUS
while true; do
  mapfile -t training_gpus < <(available_training_gpus)
  if (( ${#training_gpus[@]} >= 3 )); then
    break
  fi
  sleep 30
done
evaluation_gpu=${training_gpus[0]}
write_state TRAINING_THREE_SEEDS_IN_PARALLEL
echo "V5-A three-seed GPU mapping: 20260910=${training_gpus[0]} 20260911=${training_gpus[1]} 20260912=${training_gpus[2]}"
train_seed 20260910 "${training_gpus[0]}" & first=$!
train_seed 20260911 "${training_gpus[1]}" & second=$!
train_seed 20260912 "${training_gpus[2]}" & third=$!
set +e
wait "$first"; first_status=$?
wait "$second"; second_status=$?
wait "$third"; third_status=$?
set -e
if (( first_status != 0 || second_status != 0 || third_status != 0 )); then
  write_state ERROR_THREE_TRAINING_SEEDS
  echo "V5-A training exit codes: $first_status $second_status $third_status" >&2
  exit 5
fi

write_state SELECTING_ON_CONSUMED_DEVELOPMENT_OBJECTIVE
.venv/bin/python -m src.evaluation.v5_tail_cost_decision select \
  --run seed20260910 "$root/v5a_tail_cost_training/train2000_seed20260910" \
  --run seed20260911 "$root/v5a_tail_cost_training/train2000_seed20260911" \
  --run seed20260912 "$root/v5a_tail_cost_training/train2000_seed20260912" \
  --config "$config" --output "$root/v5a_selected_ranker.json"
checkpoint=$(.venv/bin/python -c "import json; print(json.load(open('$root/v5a_selected_ranker.json'))['selected']['checkpoint'])")

write_state EVALUATING_CONSUMED_DEVELOPMENT_HEADROOM
mkdir -p "$root/v5a_selected_development_evaluation"
CUDA_VISIBLE_DEVICES="$evaluation_gpu" HF_HUB_OFFLINE=1 \
  .venv/bin/python -m src.evaluation.relational_ranker_evaluation \
    --data "$root/v5_development300_tail_cost_oracle.jsonl" \
    --exact-dir "$root/candidates5000_exact" \
    --checkpoint "$checkpoint" \
    --output "$root/v5a_selected_development_evaluation/details.jsonl" \
    --summary "$root/v5a_selected_development_evaluation/summary.json" \
    --model models/Qwen3-1.7B --batch-size 4 --max-length 4096 --head-hidden-size 256
.venv/bin/python -m src.evaluation.v5_tail_cost_decision decide-development \
  --summary "$root/v5a_selected_development_evaluation/summary.json" \
  --checkpoint-metadata "$checkpoint/training_metadata.json" \
  --config "$config" --output "$root/v5a_development_headroom_decision.json"
decision=$(.venv/bin/python -c "import json; print(json.load(open('$root/v5a_development_headroom_decision.json'))['decision'])")
write_state "$decision"

