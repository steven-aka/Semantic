#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${1:-}" != "--execute" ]]; then
  echo "V10 is review-locked. This script has not started training."
  echo "Explicit execution is required; the protocol file must also have APPROVED_TO_RUN status:"
  echo "  bash scripts/64_launch_v10_mask_value_probe.sh --execute <physical_gpu_index>"
  exit 64
fi

physical_gpu="${2:?physical GPU index is required}"
config="configs/v10_mask_value_probe.json"

.venv/bin/python - "$config" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text())
if config["status"] != "APPROVED_TO_RUN":
    raise SystemExit(
        f"refusing to train while protocol status is {config['status']!r}; expected 'APPROVED_TO_RUN'"
    )
checks = {
    config["data"]["train"]: config["data"]["train_sha256"],
    config["data"]["development"]: config["data"]["development_sha256"],
    config["data"]["train_cache"]: config["data"]["train_cache_sha256"],
    config["data"]["development_cache"]: config["data"]["development_cache_sha256"],
    str(Path(config["initialization"]["checkpoint"]) / "training_metadata.json"):
        config["initialization"]["checkpoint_metadata_sha256"],
    str(Path(config["initialization"]["checkpoint"]) / "sequential_head.pt"):
        config["initialization"]["sequential_head_sha256"],
    **config["preflight"]["frozen_artifact_sha256"],
    **config["implementation_sha256"],
}
for name, expected in checks.items():
    path = Path(name)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"hash mismatch for {name}: {actual} != {expected}")
print(f"preflight hashes PASS ({len(checks)} artifacts)")
PY

output="results/v2_rank_then_cut/v10_mask_value_probe_seed20260912"
if [[ -e "$output/training_history.jsonl" || -e "$output/mask_value_head.pt" ]]; then
  echo "refusing to overwrite an existing V10 run in $output" >&2
  exit 65
fi
mkdir -p "$output"

.venv/bin/python - "$physical_gpu" "$config" "$output" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[3], "launch_metadata.json").write_text(json.dumps({
    "physical_gpu": int(sys.argv[1]),
    "config": sys.argv[2],
    "config_sha256_at_launch": __import__("hashlib").sha256(Path(sys.argv[2]).read_bytes()).hexdigest(),
    "launched_at_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
PY

source scripts/cuda_env.sh
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export TOKENIZERS_PARALLELISM=false

.venv/bin/python -u -m src.training.train_v10_mask_value_probe \
  --protocol-config "$config" \
  --train-data results/v2_rank_then_cut/v10_train3163_mask_value.jsonl \
  --validation-data results/v2_rank_then_cut/v10_development300_mask_value_full.jsonl \
  --exact-dir results/v2_rank_then_cut/candidates5000_exact \
  --checkpoint results/v2_rank_then_cut/v8_one_run_seed20260912/checkpoints/step250 \
  --train-cache results/v2_rank_then_cut/v10_train3163_v8_embeddings.pt \
  --validation-cache results/v2_rank_then_cut/v10_development300_v8_embeddings.pt \
  --output-dir "$output" \
  --model models/Qwen3-4B \
  --model-dim 512 \
  --max-sequence-length 512 \
  --max-optimizer-steps 400 \
  --batch-size 16 \
  --learning-rate 0.0002 \
  --weight-decay 0.01 \
  --warmup-steps 40 \
  --mask-layers 2 \
  --mask-heads 8 \
  --dropout 0.1 \
  --evaluation-mask-chunk-size 256 \
  --seed 20260912 \
  --log-interval 20 \
  2>&1 | tee "$output/run.log"
