#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/cuda_env.sh
root=results/v2_rank_then_cut
model=models/Qwen3-4B
train="$root/v7_train3163_tail_cost_oracle.jsonl"
development="$root/v5_development300_tail_cost_oracle.jsonl"
manifest="$root/v7_train3163_manifest.json"
config=configs/v7_train3163_data_scale.json
state="$root/v7_scheduler_state.json"

write_state() {
  .venv/bin/python - "$state" "$1" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
path=Path(sys.argv[1]); tmp=path.with_suffix(path.suffix+'.tmp')
tmp.write_text(json.dumps({'status':sys.argv[2],'updated_at':datetime.now().astimezone().isoformat()},indent=2)+'\n')
tmp.replace(path)
PY
}

write_state WAITING_FOR_V7_DATA
while [[ ! -s "$manifest" || ! -s "$train" ]]; do sleep 15; done

write_state VERIFYING_V7_DATA
.venv/bin/python - "$manifest" "$train" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
m=json.load(open(sys.argv[1]))
assert m['selected_examples']==3163
assert m['added_train_examples']==1163
assert m['original_train_is_exact_prefix'] is True
assert m['calibration_or_final_test_used'] is False
assert m['fresh_confirmation_used'] is False
assert sha(sys.argv[2])==m['output_tail_cost_sha256']
rows=[json.loads(line) for line in open(sys.argv[2])]
assert len(rows)==3163 and len({row['example_id'] for row in rows})==3163
dev={json.loads(line)['example_id'] for line in open('results/v2_rank_then_cut/v5_development300_tail_cost_oracle.jsonl')}
assert not dev & {row['example_id'] for row in rows}
print('V7 expanded training data: PASS')
PY

available_gpus() {
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', *' '$1 != 4 && $2 >= 33000 && $3 <= 30 {print $1, $2}' \
    | sort -k2,2nr | awk '{print $1}'
}

write_state WAITING_FOR_SMOKE_GPU
while true; do
  mapfile -t smoke_gpus < <(available_gpus)
  (( ${#smoke_gpus[@]} >= 1 )) && break
  sleep 20
done
smoke_gpu=${smoke_gpus[0]}
smoke_dir="$root/v7_batch4_smoke"
mkdir -p "$smoke_dir"
.venv/bin/python - "$train" "$smoke_dir" <<'PY'
import json,sys
from pathlib import Path
rows=[json.loads(line) for line in open(sys.argv[1])]
rows=sorted(rows,key=lambda row:sum(row['packet_tokens']),reverse=True)[:16]
out=Path(sys.argv[2])
for name,part in [('train',rows[:8]),('validation',rows[8:])]:
 out.joinpath(name+'.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in part))
PY
write_state RUNNING_BATCH4_MEMORY_SMOKE
CUDA_VISIBLE_DEVICES="$smoke_gpu" HF_HUB_OFFLINE=1 \
  .venv/bin/python -m src.training.train_data_scaled_relational_ranker \
    --train-data "$smoke_dir/train.jsonl" --validation-data "$smoke_dir/validation.jsonl" \
    --model "$model" --output-dir "$smoke_dir/output" \
    --max-optimizer-steps 1 --validation-interval-steps 1 \
    --batch-size 4 --gradient-accumulation 2 --learning-rate 0.0001 \
    --max-length 4096 --seed 20260910 --head-hidden-size 256 \
    --tail-fraction 0.25 --rate-lambda 0.25 >"$smoke_dir/smoke.log" 2>&1

write_state FREEZING_V7_PROTOCOL
.venv/bin/python - "$manifest" "$train" "$development" "$config" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
manifest,train,development,output=sys.argv[1:]
implementation=[
 'src/data/build_v7_expanded_train.py','src/model/relational_packet_ranker.py',
 'src/model/tail_cost_relational_loss.py','src/training/train_data_scaled_relational_ranker.py',
 'src/search/weighted_precedence.py','src/evaluation/relational_ranker_evaluation.py',
 'src/evaluation/v7_data_scale_decision.py',
]
model_files=sorted(str(path) for path in Path('models/Qwen3-4B').iterdir() if path.is_file())
config={
 'protocol':'v7_all_attainable_training_role_coverage_test',
 'status':'final isolated test of the one-shot static relational ranking family',
 'frozen_at':'after V6 stopped and V7 batch-4 memory smoke, before formal V7 training',
 'hypothesis':'Adding every attainable example from the original frozen training candidate slice improves compositional evidence coverage enough to pass the unchanged consumed-development gate.',
 'single_scientific_change':{'train_examples':{'old':2000,'new':3163}},
 'preserved_invariants':{
  'backbone':'Qwen3-4B NF4 QLoRA','objective':'unchanged V5-A safety-tail plus rate loss',
  'decoder':'unchanged exact maximum-weight total-order subset DP','packets':12,
  'target':'frozen Qwen3-8B exact lattice','fresh_confirmation_access':False,
  'calibration_or_final_test_access':False,
 },
 'systems_change':{'micro_batch_size':4,'gradient_accumulation':2,'effective_batch_size':8,'reason':'use available VRAM while preserving effective batch size'},
 'optimization_budget':{
  'max_optimizer_steps':750,'validation_interval_steps':250,'learning_rate':0.0001,
  'reason':'match V6 optimizer exposure and checkpoint-selection opportunities despite the larger dataset',
 },
 'training':{'seeds':[20260910,20260911,20260912],'max_length':4096,'lora_rank':32,'head_hidden_size':256,'tail_fraction':0.25,'rate_lambda':0.25},
 'data':{
  'manifest':manifest,'manifest_sha256':sha(manifest),'train':train,'train_sha256':sha(train),
  'development':development,'development_sha256':sha(development),
 },
 'consumed_development_gate':{
  'fidelity_0_90_successes':'at least 282/300','all_active_trajectory_success':'at least 0.90',
  'mean_feasible_normalized_ranking_regret':'at most 0.03',
  'if_pass':'freeze a new target-blind confirmation population before any Target inference',
  'if_fail':'STOP_STATIC_ONE_SHOT_RANKING_AFTER_V7',
 },
 'model':{'path':'models/Qwen3-4B','files_sha256':{path:sha(path) for path in model_files}},
 'implementation_sha256':{path:sha(path) for path in implementation},
}
Path(output).write_text(json.dumps(config,indent=2)+'\n')
print('V7 config sha256',sha(output))
PY

write_state VERIFYING_FROZEN_V7
.venv/bin/python - "$config" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
c=json.load(open(sys.argv[1]))
for path,expected in c['implementation_sha256'].items(): assert sha(path)==expected,(path,sha(path),expected)
for key in ('manifest','train','development'): assert sha(c['data'][key])==c['data'][key+'_sha256']
for path,expected in c['model']['files_sha256'].items(): assert sha(path)==expected,(path,sha(path),expected)
print('V7 frozen hashes: PASS')
PY

train_seed() {
  local seed=$1
  local gpu=$2
  local out="$root/v7_4b_training/train3163_seed${seed}"
  mkdir -p "$root/v7_4b_training/logs"
  [[ -s "$out/training_metadata.json" ]] && return
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 \
    .venv/bin/python -m src.training.train_data_scaled_relational_ranker \
      --train-data "$train" --validation-data "$development" --model "$model" --output-dir "$out" \
      --max-optimizer-steps 750 --validation-interval-steps 250 \
      --batch-size 4 --gradient-accumulation 2 --learning-rate 0.0001 --max-length 4096 \
      --seed "$seed" --head-hidden-size 256 --tail-fraction 0.25 --rate-lambda 0.25 \
      >"$root/v7_4b_training/logs/train3163_seed${seed}.log" 2>&1
}

write_state WAITING_FOR_THREE_TRAIN_GPUS
while true; do
  mapfile -t training_gpus < <(available_gpus)
  (( ${#training_gpus[@]} >= 3 )) && break
  sleep 20
done
write_state TRAINING_ALL_THREE_SEEDS_IN_PARALLEL
echo "V7 GPU mapping: 20260910=${training_gpus[0]} 20260911=${training_gpus[1]} 20260912=${training_gpus[2]}"
train_seed 20260910 "${training_gpus[0]}" & p1=$!
train_seed 20260911 "${training_gpus[1]}" & p2=$!
train_seed 20260912 "${training_gpus[2]}" & p3=$!
set +e
wait "$p1"; s1=$?; wait "$p2"; s2=$?; wait "$p3"; s3=$?
set -e
if (( s1 || s2 || s3 )); then write_state ERROR_V7_TRAINING; exit 5; fi

write_state SELECTING_ON_CONSUMED_DEVELOPMENT_OBJECTIVE
.venv/bin/python -m src.evaluation.v7_data_scale_decision select \
  --run seed20260910 "$root/v7_4b_training/train3163_seed20260910" \
  --run seed20260911 "$root/v7_4b_training/train3163_seed20260911" \
  --run seed20260912 "$root/v7_4b_training/train3163_seed20260912" \
  --config "$config" --output "$root/v7_selected_ranker.json"
checkpoint=$(.venv/bin/python -c "import json; print(json.load(open('$root/v7_selected_ranker.json'))['selected']['checkpoint'])")

write_state EVALUATING_CONSUMED_DEVELOPMENT_GATE
mkdir -p "$root/v7_selected_development_evaluation"
CUDA_VISIBLE_DEVICES="${training_gpus[0]}" HF_HUB_OFFLINE=1 \
  .venv/bin/python -m src.evaluation.relational_ranker_evaluation \
    --data "$development" --exact-dir "$root/candidates5000_exact" --checkpoint "$checkpoint" \
    --output "$root/v7_selected_development_evaluation/details.jsonl" \
    --summary "$root/v7_selected_development_evaluation/summary.json" \
    --model "$model" --batch-size 4 --max-length 4096 --head-hidden-size 256
.venv/bin/python -m src.evaluation.v7_data_scale_decision decide-development \
  --summary "$root/v7_selected_development_evaluation/summary.json" \
  --checkpoint-metadata "$checkpoint/training_metadata.json" --config "$config" \
  --output "$root/v7_development_headroom_decision.json"
decision=$(.venv/bin/python -c "import json; print(json.load(open('$root/v7_development_headroom_decision.json'))['decision'])")
write_state "$decision"
