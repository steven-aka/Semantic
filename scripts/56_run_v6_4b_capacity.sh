#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/cuda_env.sh
root=results/v2_rank_then_cut
model=models/Qwen3-4B
config=configs/v6_4b_capacity_ranking.json
state="$root/v6_scheduler_state.json"
export HF_HOME=/backup01/zhangyihan/hf_cache
export HF_XET_CACHE=/backup01/zhangyihan/hf_cache/xet

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

write_state DOWNLOADING_QWEN3_4B
HF_XET_HIGH_PERFORMANCE=1 .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3-4B',
    local_dir='models/Qwen3-4B',
    allow_patterns=['*.json','*.safetensors','tokenizer*','*.model','*.txt'],
)
PY

write_state VALIDATING_QWEN3_4B
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root=Path('models/Qwen3-4B')
index=json.loads((root/'model.safetensors.index.json').read_text())
shards=sorted(set(index['weight_map'].values()))
if shards != [f'model-{i:05d}-of-00003.safetensors' for i in range(1,4)]:
    raise SystemExit(f'unexpected shards: {shards}')
for shard in shards:
    path=root/shard
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f'missing model shard: {path}')
config=json.loads((root/'config.json').read_text())
if config.get('hidden_size') != 2560 or config.get('num_hidden_layers') != 36:
    raise SystemExit('Qwen3-4B architecture metadata changed')
print('Qwen3-4B local snapshot validation: PASS')
PY

available_training_gpus() {
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
    --format=csv,noheader,nounits \
    | awk -F', *' '$1 != 4 && $2 >= 35000 && $3 <= 20 {print $1, $2}' \
    | sort -k2,2nr | awk '{print $1}'
}

write_state WAITING_FOR_SMOKE_GPU
while true; do
  mapfile -t smoke_gpus < <(available_training_gpus)
  (( ${#smoke_gpus[@]} >= 1 )) && break
  sleep 30
done
smoke_gpu=${smoke_gpus[0]}
smoke_dir="$root/v6_4b_smoke"
mkdir -p "$smoke_dir"
.venv/bin/python - <<'PY'
import json
from pathlib import Path
rows=[json.loads(line) for line in open('results/v2_rank_then_cut/v5_train2000_tail_cost_oracle.jsonl')]
rows=sorted(rows,key=lambda row:sum(row['packet_tokens']),reverse=True)[:16]
for name,part in [('train',rows[:8]),('validation',rows[8:])]:
    Path(f'results/v2_rank_then_cut/v6_4b_smoke/{name}.jsonl').write_text(
        ''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in part)
    )
PY
write_state RUNNING_4B_MEMORY_SMOKE
CUDA_VISIBLE_DEVICES="$smoke_gpu" HF_HUB_OFFLINE=1 \
  .venv/bin/python -m src.training.train_capacity_relational_ranker \
    --train-data "$smoke_dir/train.jsonl" \
    --validation-data "$smoke_dir/validation.jsonl" \
    --model "$model" --output-dir "$smoke_dir/output" \
    --epochs 1 --batch-size 2 --gradient-accumulation 4 \
    --learning-rate 0.0001 --max-length 4096 --seed 20260910 \
    --head-hidden-size 256 --tail-fraction 0.25 --rate-lambda 0.25 \
    >"$smoke_dir/smoke.log" 2>&1

write_state FREEZING_V6_PROTOCOL
.venv/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
implementation=[
 'src/model/relational_packet_ranker.py',
 'src/model/tail_cost_relational_loss.py',
 'src/training/train_capacity_relational_ranker.py',
 'src/search/weighted_precedence.py',
 'src/evaluation/relational_ranker_evaluation.py',
 'src/evaluation/v6_capacity_decision.py',
]
model_files=sorted(str(path) for path in Path('models/Qwen3-4B').iterdir() if path.is_file())
config={
 'protocol':'v6_qwen3_4b_capacity_relational_ranking',
 'frozen_at':'after completed V5-A consumed-development decoder/distribution audits and 4B memory smoke, before formal V6 training or any new confirmation Target inference',
 'status':'single controlled backbone-capacity test after STOP_V5A_WITHOUT_FRESH_CONFIRM',
 'scientific_hypothesis':'The high-confidence, three-seed-consistent precedence errors concentrated in compositional evidence require a stronger semantic representation; increasing only the learner backbone from Qwen3-1.7B to Qwen3-4B can improve critical-edge direction generalization while retaining V5-A rate efficiency.',
 'parent_decision':{'path':'results/v2_rank_then_cut/v5a_development_headroom_decision.json','decision':'STOP_V5A_WITHOUT_FRESH_CONFIRM'},
 'diagnostics':{
  'decoder_alignment':'results/v2_rank_then_cut/v5a_decoder_alignment_diagnostic/summary.json',
  'decoder_alignment_sha256':sha('results/v2_rank_then_cut/v5a_decoder_alignment_diagnostic/summary.json'),
  'three_seed_tail':'results/v2_rank_then_cut/v5a_three_seed_tail_diagnostic/summary.json',
  'three_seed_tail_sha256':sha('results/v2_rank_then_cut/v5a_three_seed_tail_diagnostic/summary.json'),
  'raw_correct_edges_sacrificed_by_decoder':0,
  'failed_repair_edges_raw_wrong':32,
  'failed_repair_edges_unanimous_wrong':30,
  'failed_repair_median_absolute_margin':2.296875,
  'wikitables_composition_failures':17,
 },
 'single_controlled_change':{'old_backbone':'Qwen3-1.7B NF4 QLoRA','new_backbone':'Qwen3-4B NF4 QLoRA'},
 'preserved_invariants':{
  'packets':'12 source-preserving lossless binary packets',
  'relational_head':'same antisymmetric ordered-pair MLP with hidden size 256',
  'objective':'same V5-A top-25% weighted safety CVaR plus 0.25 rate-dominance loss',
  'decoder':'same exact maximum-weight total-order subset DP',
  'target':'frozen Qwen3-8B exact lattice',
  'training_data':'unchanged train2000',
  'development_data':'unchanged consumed development300',
  'cutoff_training':False,
  'fresh_confirmation_access':False,
  'calibration_or_final_test_access':False,
 },
 'data':{
  'train':'results/v2_rank_then_cut/v5_train2000_tail_cost_oracle.jsonl',
  'train_sha256':sha('results/v2_rank_then_cut/v5_train2000_tail_cost_oracle.jsonl'),
  'development':'results/v2_rank_then_cut/v5_development300_tail_cost_oracle.jsonl',
  'development_sha256':sha('results/v2_rank_then_cut/v5_development300_tail_cost_oracle.jsonl'),
  'v4_fresh300':'locked and not reused',
  'future_fresh_confirmation':'unassigned unless the consumed-development gate passes',
 },
 'model':{'path':'models/Qwen3-4B','files_sha256':{path:sha(path) for path in model_files}},
 'training':{
  'seeds':[20260910,20260911,20260912],
  'epochs':3,
  'batch_size':2,
  'gradient_accumulation':4,
  'effective_batch_size':8,
  'learning_rate':0.0001,
  'max_length':4096,
  'lora_rank':32,
  'head_hidden_size':256,
  'tail_fraction':0.25,
  'rate_lambda':0.25,
  'selection':'minimum consumed-development combined tail-cost objective',
  'epoch_cap_reason':'all V4 and V5-A seeds selected epoch 1 or 2 and overfit afterward',
 },
 'consumed_development_headroom_gate':{
  'fidelity_0_90_successes':'at least 282/300',
  'all_active_trajectory_success':'at least 0.90',
  'mean_feasible_normalized_ranking_regret':'at most 0.03',
  'if_pass':'freeze a new target-blind confirmation population before Target inference',
  'if_fail':'STOP_V6_WITHOUT_FRESH_CONFIRM',
 },
 'implementation_sha256':{path:sha(path) for path in implementation},
}
Path('configs/v6_4b_capacity_ranking.json').write_text(json.dumps(config,indent=2)+'\n')
print('V6 config sha256',sha('configs/v6_4b_capacity_ranking.json'))
PY

write_state VERIFYING_FROZEN_V6
.venv/bin/python - "$config" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
c=json.loads(Path(sys.argv[1]).read_text())
for path,expected in c['implementation_sha256'].items():
    assert sha(path)==expected,(path,sha(path),expected)
for role in ('train','development'):
    assert sha(c['data'][role])==c['data'][role+'_sha256']
for path,expected in c['model']['files_sha256'].items():
    assert sha(path)==expected,(path,sha(path),expected)
print('V6 frozen hashes: PASS')
PY

train_seed() {
  local seed=$1 gpu=$2
  local out="$root/v6_4b_training/train2000_seed${seed}"
  mkdir -p "$root/v6_4b_training/logs"
  if [[ -s "$out/training_metadata.json" ]]; then
    echo "reuse complete V6 seed $seed"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 \
    .venv/bin/python -m src.training.train_capacity_relational_ranker \
      --train-data "$root/v5_train2000_tail_cost_oracle.jsonl" \
      --validation-data "$root/v5_development300_tail_cost_oracle.jsonl" \
      --model "$model" --output-dir "$out" \
      --epochs 3 --batch-size 2 --gradient-accumulation 4 \
      --learning-rate 0.0001 --max-length 4096 --seed "$seed" \
      --head-hidden-size 256 --tail-fraction 0.25 --rate-lambda 0.25 \
      >"$root/v6_4b_training/logs/train2000_seed${seed}.log" 2>&1
}

write_state WAITING_FOR_TWO_TRAIN_GPUS
while true; do
  mapfile -t training_gpus < <(available_training_gpus)
  (( ${#training_gpus[@]} >= 2 )) && break
  sleep 30
done
first_gpu=${training_gpus[0]}
second_gpu=${training_gpus[1]}
write_state TRAINING_FIRST_TWO_SEEDS_IN_PARALLEL
echo "V6 GPU mapping: 20260910=$first_gpu 20260911=$second_gpu"
train_seed 20260910 "$first_gpu" & first=$!
train_seed 20260911 "$second_gpu" & second=$!
set +e
wait "$first"; first_status=$?
wait "$second"; second_status=$?
set -e
if (( first_status != 0 || second_status != 0 )); then
  write_state ERROR_FIRST_TWO_TRAINING_SEEDS
  exit 5
fi

write_state TRAINING_THIRD_SEED
train_seed 20260912 "$first_gpu"

write_state SELECTING_ON_CONSUMED_DEVELOPMENT_OBJECTIVE
.venv/bin/python -m src.evaluation.v6_capacity_decision select \
  --run seed20260910 "$root/v6_4b_training/train2000_seed20260910" \
  --run seed20260911 "$root/v6_4b_training/train2000_seed20260911" \
  --run seed20260912 "$root/v6_4b_training/train2000_seed20260912" \
  --config "$config" --output "$root/v6_selected_ranker.json"
checkpoint=$(.venv/bin/python -c "import json; print(json.load(open('$root/v6_selected_ranker.json'))['selected']['checkpoint'])")

write_state EVALUATING_CONSUMED_DEVELOPMENT_HEADROOM
mkdir -p "$root/v6_selected_development_evaluation"
CUDA_VISIBLE_DEVICES="$first_gpu" HF_HUB_OFFLINE=1 \
  .venv/bin/python -m src.evaluation.relational_ranker_evaluation \
    --data "$root/v5_development300_tail_cost_oracle.jsonl" \
    --exact-dir "$root/candidates5000_exact" \
    --checkpoint "$checkpoint" \
    --output "$root/v6_selected_development_evaluation/details.jsonl" \
    --summary "$root/v6_selected_development_evaluation/summary.json" \
    --model "$model" --batch-size 2 --max-length 4096 --head-hidden-size 256
.venv/bin/python -m src.evaluation.v6_capacity_decision decide-development \
  --summary "$root/v6_selected_development_evaluation/summary.json" \
  --checkpoint-metadata "$checkpoint/training_metadata.json" \
  --config "$config" --output "$root/v6_development_headroom_decision.json"
decision=$(.venv/bin/python -c "import json; print(json.load(open('$root/v6_development_headroom_decision.json'))['decision'])")
write_state "$decision"
