#!/usr/bin/env bash
set -euo pipefail
PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}
"$PROJECT_PYTHON" -m src.evaluation.v0_3_gate \
  --config configs/v0_5_gate.json \
  --manifest results/v0_5/data_manifest.json \
  --examples data/units/hotpot_v0_5_candidates.jsonl \
  --raw data/raw/hotpot_validation.parquet \
  --baseline results/v0_5/full_context.jsonl \
  --packet-dir data/packets_v0_5 \
  --coverage results/v0_5/fact_coverage.jsonl \
  --raw-exact-dir results/v0_5/exact_search_raw \
  --fact-exact-dir results/v0_5/exact_search \
  --tokenizer models/Qwen3-8B \
  --output results/v0_5/gate_result.json \
  --output-dir results/v0_5
