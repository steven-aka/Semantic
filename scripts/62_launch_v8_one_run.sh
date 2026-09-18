#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/cuda_env.sh
root=results/v2_rank_then_cut
output="$root/v8_one_run_seed20260912"
log="$root/v8_one_run_seed20260912.log"
state="$root/v8_one_run_state.json"
gpu=${V8_GPU:-2}

.venv/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
c=json.load(open('configs/v8_one_run_sequential.json'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p,h in c['implementation_sha256'].items(): assert sha(p)==h,(p,sha(p),h)
for key in ('train','train_manifest','development','development_manifest'):
 assert sha(c['data'][key])==c['data'][key+'_sha256'],key
for p,h in c['model']['files_sha256'].items(): assert sha(p)==h,p
assert c['data']['locked_roles_used'] is False
print('V8 frozen hashes: PASS')
PY

if [[ -s "$output/training_metadata.json" ]]; then
  echo "V8 already complete: $output"
  exit 0
fi
if [[ -s "$root/v8_one_run.pid" ]] && kill -0 "$(cat "$root/v8_one_run.pid")" 2>/dev/null; then
  echo "V8 already running with PID $(cat "$root/v8_one_run.pid")"
  exit 0
fi

.venv/bin/python - "$state" "$gpu" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
 'status':'RUNNING_FORMAL_V8_ONE_RUN','physical_gpu':int(sys.argv[2]),
 'started_at':datetime.now().astimezone().isoformat(),
 'config':'configs/v8_one_run_sequential.json',
},indent=2)+'\n')
PY

nohup setsid bash -lc "
  cd '$PWD'
  echo \$\$ > '$root/v8_one_run.pid'
  source scripts/cuda_env.sh
  export CUDA_VISIBLE_DEVICES='$gpu'
  export HF_HUB_OFFLINE=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  exec .venv/bin/python -m src.training.train_v8_sequential_policy \
    --train-data '$root/v8_train3163_history_supervision.jsonl' \
    --validation-data '$root/v8_development300_history_supervision.jsonl' \
    --exact-dir '$root/candidates5000_exact' \
    --model models/Qwen3-4B --output-dir '$output' \
    --max-optimizer-steps 750 --validation-interval-steps 250 \
    --batch-size 8 --validation-batch-size 8 --gradient-accumulation 1 \
    --lora-learning-rate 0.00002 --head-learning-rate 0.0002 \
    --weight-decay 0.01 --warmup-steps 50 --progress-weight 0.2 \
    --max-sequence-length 512 --model-dim 512 --beam-width 8 --seed 20260912
" >"$log" 2>&1 </dev/null &
pid=$!
echo "Launched formal V8 PID=$pid physical_gpu=$gpu log=$log"
