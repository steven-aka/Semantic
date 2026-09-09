#!/usr/bin/env bash
set -euo pipefail

PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}
RAW_INPUT=${1:-data/raw/hotpot_validation.parquet}
N_EXAMPLES=${N_EXAMPLES:-10}
N_UNITS=${N_UNITS:-5}
CANDIDATE_LIMIT=${CANDIDATE_LIMIT:-100}
export HF_HOME=${HF_HOME:-"$PWD/.cache/huggingface"}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}

"$PROJECT_PYTHON" -m src.data.normalize_hotpot \
  --input "$RAW_INPUT" \
  --output data/normalized/hotpot_validation.jsonl \
  --limit "$CANDIDATE_LIMIT"
"$PROJECT_PYTHON" -m src.data.segment \
  --input data/normalized/hotpot_validation.jsonl \
  --output data/units/hotpot_validation.jsonl \
  --tokenizer Qwen/Qwen3-8B
"$PROJECT_PYTHON" -m src.data.select_pilot \
  --input data/units/hotpot_validation.jsonl \
  --output data/units/hotpot_exact_pilot.jsonl \
  --units "$N_UNITS" \
  --limit "$N_EXAMPLES"
"$PROJECT_PYTHON" -m src.evaluation.data_audit \
  --input data/units/hotpot_exact_pilot.jsonl \
  --output results/data_audit_smoke.json \
  --tokenizer Qwen/Qwen3-8B
