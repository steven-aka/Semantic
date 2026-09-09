#!/usr/bin/env bash
set -euo pipefail
. scripts/cuda_env.sh

PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}

"$PROJECT_PYTHON" -m src.evaluation.verify_v0_2
"$PROJECT_PYTHON" -m unittest discover -s tests -v
"$PROJECT_PYTHON" -m py_compile \
  src/data/fact_atomic_pilot.py \
  src/evaluation/fact_coverage.py \
  src/evaluation/run_fact_atomic_target.py \
  src/evaluation/verify_v0_2.py \
  src/search/exact_frontier.py \
  src/search/best_nested_chain.py
bash -n scripts/*.sh
