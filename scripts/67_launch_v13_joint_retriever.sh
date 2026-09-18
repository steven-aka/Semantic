#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

physical_gpu="${1:-4}"
config=configs/v13_joint_retriever.json
output=results/v2_rank_then_cut/v13_joint_retriever_seed20260918

.venv/bin/python - "$config" <<'PY'
import hashlib, json, sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
if config["status"] != "APPROVED_TO_RUN":
    raise SystemExit("protocol is not approved")
for path, expected in config["artifacts_sha256"].items():
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"hash mismatch: {path}")
print(f"preflight hashes PASS ({len(config['artifacts_sha256'])} artifacts)")
PY

.venv/bin/python - "$output" <<'PY'
from pathlib import Path
import sys

output = Path(sys.argv[1])
for forbidden in (output / "history.jsonl", output / "selected_checkpoint.json", output / "checkpoints"):
    if forbidden.exists():
        raise SystemExit(f"refusing to overwrite formal artifact: {forbidden}")
PY

mkdir -p "$output"
source scripts/cuda_env.sh
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv/bin/python -u -m src.training.train_v13_joint_retriever \
  --protocol-config "$config" \
  --train-data results/v2_rank_then_cut/v13_internal_train2863_mask_value.jsonl \
  --validation-data results/v2_rank_then_cut/v13_internal_validation300_mask_value.jsonl \
  --checkpoint results/v2_rank_then_cut/v8_one_run_seed20260912/checkpoints/step250 \
  --initial-head results/v2_rank_then_cut/v12_high_fidelity_retriever_seed20260918/mask_retrieval_head.pt \
  --output-dir "$output" --model models/Qwen3-4B --model-dim 512 \
  --max-sequence-length 512 --steps 300 --validation-interval 100 \
  --batch-size 16 --gradient-accumulation 1 --lora-lr 1e-5 --head-lr 1e-4 \
  --seed 20260918 2>&1 | tee "$output/run.log"

selected="$(.venv/bin/python - "$output/selected_checkpoint.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["checkpoint"])
PY
)"

.venv/bin/python -u -m src.data.cache_v10_encoder_embeddings \
  --data results/v2_rank_then_cut/v10_development300_mask_value_full.jsonl \
  --checkpoint "$selected" \
  --output "$output/development300_embeddings.pt" \
  --manifest "$output/development300_embeddings_manifest.json" \
  --model models/Qwen3-4B --model-dim 512 --max-sequence-length 512 --batch-size 16 \
  2>&1 | tee -a "$output/run.log"

.venv/bin/python -u -m src.evaluation.v11_mask_retrieval_audit \
  --data results/v2_rank_then_cut/v10_development300_mask_value_full.jsonl \
  --cache "$output/development300_embeddings.pt" \
  --head "$selected/mask_retrieval_head.pt" \
  --v8-details results/v2_rank_then_cut/v8_one_run_seed20260912/checkpoints/step250/development_details.jsonl \
  --output "$output/development_details.jsonl" \
  --summary "$output/development_summary.json" --batch-size 16 --mask-chunk-size 512 \
  2>&1 | tee -a "$output/run.log"
