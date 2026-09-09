#!/usr/bin/env bash
set -euo pipefail
. scripts/cuda_env.sh

PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}
"$PROJECT_PYTHON" -m src.data.multihop_retrieval \
  --config configs/retrieval_r0.json \
  --raw data/raw/hotpot_validation.parquet \
  --output-dir results/retrieval_r0 \
  --candidate-output data/units/hotpot_v0_4_candidates.jsonl \
  --tokenizer models/Qwen3-8B
