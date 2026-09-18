#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/cuda_env.sh
root=results/v2_rank_then_cut
state="$root/v4_scheduler_state.json"
write_state() {
  .venv/bin/python - "$state" "$1" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
path=Path(sys.argv[1]); value={"status":sys.argv[2],"updated_at":datetime.now().astimezone().isoformat()}
tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2)+'\n'); tmp.replace(path)
PY
}

write_state WAITING_EXACT_500
restart_attempts=(0 0)
while true; do
  complete=$(find "$root/v4_candidates500_exact" -maxdepth 1 -name '*.jsonl' -type f | wc -l)
  [[ "$complete" == 500 ]] && break
  mapfile -t shard_counts < <(.venv/bin/python - <<'PY'
from pathlib import Path
from src.data.schemas import read_jsonl
rows=list(read_jsonl('data/units/qampari_rank_v4_candidates500.jsonl'))
out=Path('results/v2_rank_then_cut/v4_candidates500_exact')
for shard in range(2):
    print(sum((out/f"{row['example_id']}.jsonl").is_file() for index,row in enumerate(rows) if index%2==shard))
PY
)
  for shard in 0 1; do
    if (( shard_counts[$shard] == 250 )); then
      continue
    fi
    running=$(pgrep -f "[s]rc.search.atomic_exact_search.*v4_candidates500_exact.*--shard-index $shard --num-shards 2" | head -1 || true)
    if [[ -n "$running" ]]; then
      continue
    fi
    if (( restart_attempts[$shard] >= 2 )); then
      write_state ERROR_EXACT_RESTART_LIMIT
      echo "shard $shard stopped after two bounded operational restarts" >&2
      exit 3
    fi
    restart_attempts[$shard]=$(( restart_attempts[$shard] + 1 ))
    write_state "RESTARTING_EXACT_SHARD_${shard}_ATTEMPT_${restart_attempts[$shard]}"
    bash scripts/46_launch_v4_candidate_exact.sh
  done
  write_state WAITING_EXACT_500
  sleep 60
done

write_state VALIDATING_AND_FREEZING_FRESH300
bash scripts/48_finalize_v4_rank_confirm.sh

wait_gpu() {
  local gpu=$1
  if [[ "$gpu" == 4 ]]; then
    echo "physical GPU4 is excluded" >&2
    exit 4
  fi
  while true; do
    read -r free util < <(nvidia-smi -i "$gpu" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
    if (( free >= 16000 && util <= 20 )); then
      return
    fi
    sleep 30
  done
}

train_seed() {
  local seed=$1 gpu=$2
  local out="$root/v4_relational_training/train2000_seed${seed}"
  if [[ -s "$out/training_metadata.json" ]]; then
    echo "reuse complete V4 seed $seed"
    return
  fi
  mkdir -p "$root/v4_relational_training/logs"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 \
    .venv/bin/python -m src.training.train_relational_boundary_ranker \
      --train-data "$root/v3_train2000_boundary_oracle.jsonl" \
      --validation-data "$root/v3_development300_boundary_oracle.jsonl" \
      --model models/Qwen3-1.7B \
      --output-dir "$out" \
      --epochs 5 --batch-size 1 --gradient-accumulation 8 \
      --learning-rate 0.0001 --max-length 4096 \
      --seed "$seed" --head-hidden-size 256 \
      >"$root/v4_relational_training/logs/train2000_seed${seed}.log" 2>&1
}

available_training_gpus() {
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
    --format=csv,noheader,nounits \
    | awk -F', *' '$1 != 4 && $2 >= 16000 && $3 <= 20 {print $1, $2}' \
    | sort -k2,2nr | awk '{print $1}'
}

write_state WAITING_FOR_THREE_TRAIN_GPUS
while true; do
  mapfile -t training_gpus < <(available_training_gpus)
  if (( ${#training_gpus[@]} >= 3 )); then
    sleep 30
    mapfile -t stable_gpus < <(available_training_gpus)
    if [[ " ${stable_gpus[*]} " == *" ${training_gpus[0]} "* \
       && " ${stable_gpus[*]} " == *" ${training_gpus[1]} "* \
       && " ${stable_gpus[*]} " == *" ${training_gpus[2]} "* ]]; then
      break
    fi
  fi
  sleep 30
done
evaluation_gpu=${training_gpus[0]}
write_state TRAINING_THREE_SEEDS_IN_PARALLEL
echo "V4 three-seed GPU mapping: 20260910=${training_gpus[0]} 20260911=${training_gpus[1]} 20260912=${training_gpus[2]}"
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
  echo "V4 training exit codes: $first_status $second_status $third_status" >&2
  exit 5
fi

write_state SELECTING_ON_CONSUMED_DEVELOPMENT
.venv/bin/python -m src.evaluation.v4_relational_decision select \
  --run seed20260910 "$root/v4_relational_training/train2000_seed20260910" \
  --run seed20260911 "$root/v4_relational_training/train2000_seed20260911" \
  --run seed20260912 "$root/v4_relational_training/train2000_seed20260912" \
  --config configs/v4_relational_precedence_ranking.json \
  --output "$root/v4_selected_ranker.json"
checkpoint=$(.venv/bin/python -c "import json; print(json.load(open('$root/v4_selected_ranker.json'))['selected']['checkpoint'])")

write_state EVALUATING_CONSUMED_DEVELOPMENT_HEADROOM
mkdir -p "$root/v4_selected_development_evaluation"
wait_gpu "$evaluation_gpu"
CUDA_VISIBLE_DEVICES="$evaluation_gpu" HF_HUB_OFFLINE=1 \
  .venv/bin/python -m src.evaluation.relational_ranker_evaluation \
    --data "$root/v3_development300_boundary_oracle.jsonl" \
    --exact-dir "$root/candidates5000_exact" \
    --checkpoint "$checkpoint" \
    --output "$root/v4_selected_development_evaluation/details.jsonl" \
    --summary "$root/v4_selected_development_evaluation/summary.json" \
    --model models/Qwen3-1.7B --batch-size 4 --max-length 4096 --head-hidden-size 256
.venv/bin/python -m src.evaluation.v4_relational_decision decide \
  --stage development \
  --summary "$root/v4_selected_development_evaluation/summary.json" \
  --checkpoint-metadata "$checkpoint/training_metadata.json" \
  --config configs/v4_relational_precedence_ranking.json \
  --output "$root/v4_development_headroom_decision.json"
go_fresh=$(.venv/bin/python -c "import json; print(int(json.load(open('$root/v4_development_headroom_decision.json'))['passed']))")
if [[ "$go_fresh" != 1 ]]; then
  write_state STOP_V4_WITHOUT_FRESH_CONFIRM
  exit 0
fi

write_state ONE_SHOT_FRESH_CONFIRM_EVALUATION
mkdir -p "$root/v4_fresh_confirm_evaluation"
wait_gpu "$evaluation_gpu"
CUDA_VISIBLE_DEVICES="$evaluation_gpu" HF_HUB_OFFLINE=1 \
  .venv/bin/python -m src.evaluation.relational_ranker_evaluation \
    --data "$root/v4_rank_confirm300_boundary_oracle.jsonl" \
    --exact-dir "$root/v4_candidates500_exact" \
    --checkpoint "$checkpoint" \
    --output "$root/v4_fresh_confirm_evaluation/details.jsonl" \
    --summary "$root/v4_fresh_confirm_evaluation/summary.json" \
    --model models/Qwen3-1.7B --batch-size 4 --max-length 4096 --head-hidden-size 256
.venv/bin/python -m src.evaluation.v4_relational_decision decide \
  --stage fresh \
  --summary "$root/v4_fresh_confirm_evaluation/summary.json" \
  --checkpoint-metadata "$checkpoint/training_metadata.json" \
  --config configs/v4_relational_precedence_ranking.json \
  --output "$root/v4_fresh_confirm_decision.json"
decision=$(.venv/bin/python -c "import json; print(json.load(open('$root/v4_fresh_confirm_decision.json'))['decision'])")
write_state "$decision"
