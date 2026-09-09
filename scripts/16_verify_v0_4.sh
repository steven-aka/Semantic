#!/usr/bin/env bash
set -euo pipefail
PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}
"$PROJECT_PYTHON" -m src.evaluation.v0_3_gate \
  --config configs/v0_4_gate.json \
  --manifest results/retrieval_r1/manifest.json \
  --examples data/units/hotpot_v0_4_candidates.jsonl \
  --raw data/raw/hotpot_validation.parquet \
  --baseline results/v0_4/full_context.jsonl \
  --packet-dir data/packets_v0_4 \
  --coverage results/v0_4/fact_coverage.jsonl \
  --raw-exact-dir results/v0_4/exact_search_raw \
  --fact-exact-dir results/v0_4/exact_search \
  --tokenizer models/Qwen3-8B \
  --output results/v0_4/gate_result.json \
  --output-dir results/v0_4
