#!/usr/bin/env bash
set -euo pipefail
. scripts/cuda_env.sh

PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}
STAGE=${1:-all}

run_stage() {
  "$PROJECT_PYTHON" -m src.data.semantic_retrieval "$1" \
    --config configs/retrieval_r1.json \
    --raw data/raw/hotpot_validation.parquet \
    --output-dir results/retrieval_r1 \
    --candidate-output data/units/hotpot_v0_4_candidates.jsonl \
    --batch-size 8
}

case "$STAGE" in
  prepare) run_stage prepare ;;
  select) run_stage select ;;
  evaluate) run_stage evaluate ;;
  all)
    run_stage prepare
    run_stage select
    run_stage evaluate
    ;;
  *) echo "usage: $0 {prepare|select|evaluate|all}" >&2; exit 2 ;;
esac
