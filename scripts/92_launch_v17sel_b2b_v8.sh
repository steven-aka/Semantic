#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/cuda_env.sh
export CUDA_VISIBLE_DEVICES="${SEL_B2B_GPU:-0}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

root=results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout
out="$root/v8_run"

.venv/bin/python - <<'PY'
import json
from pathlib import Path
from src.reproducibility import sha256

root=Path('results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout')
manifest=json.loads((root/'manifest.json').read_text())
config=Path('configs/v17sel_b2b_lineage_clean_pool_gate.json')
assert manifest['protocol_sha256']==sha256(config)
assert manifest['holdout_count']==581
assert manifest['files']['v8_train']['train']['output_count'] + manifest['holdout_count'] + manifest['inner_validation_count']==3163
train=Path(manifest['files']['v8_train']['train']['output'])
assert sha256(train)==manifest['files']['v8_train']['train']['output_sha256']
validation=Path(manifest['files']['v8_train']['inner_validation']['output'])
assert sha256(validation)==manifest['files']['v8_train']['inner_validation']['output_sha256']
print('SEL-B2B V8 lineage preflight passed')
PY

if [[ -e "$out/history.jsonl" || -e "$out/checkpoints" ]]; then
  echo "Refusing to overwrite existing lineage-clean V8 run: $out" >&2
  exit 65
fi
mkdir -p "$out"

exec .venv/bin/python -u -m src.training.train_v8_sequential_policy \
  --train-data "$root/data/v8_train_clean.jsonl" \
  --validation-data "$root/data/v8_train_inner_validation.jsonl" \
  --exact-dir results/v2_rank_then_cut/candidates5000_exact \
  --model models/Qwen3-4B \
  --output-dir "$out" \
  --max-optimizer-steps 750 \
  --validation-interval-steps 250 \
  --batch-size 8 \
  --validation-batch-size 8 \
  --gradient-accumulation 1 \
  --lora-learning-rate 0.00002 \
  --head-learning-rate 0.0002 \
  --weight-decay 0.01 \
  --warmup-steps 50 \
  --progress-weight 0.2 \
  --max-sequence-length 512 \
  --model-dim 512 \
  --beam-width 8 \
  --seed 20260912
