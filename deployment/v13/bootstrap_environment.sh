#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r deployment/v13/requirements-v13.txt

.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-4B",
    local_dir="models/Qwen3-4B",
)
PY

echo "Environment and Qwen3-4B base model are ready."
echo "If CUDA shared libraries are site-specific, update scripts/cuda_env.sh."
