#!/usr/bin/env bash
set -euo pipefail
. scripts/cuda_env.sh

PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}
"$PROJECT_PYTHON" -m src.evaluation.v0_3_gate
"$PROJECT_PYTHON" -m unittest discover -s tests -v
"$PROJECT_PYTHON" -m compileall -q src tests
for script in scripts/*.sh; do bash -n "$script"; done
