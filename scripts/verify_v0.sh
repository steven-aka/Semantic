#!/usr/bin/env bash
set -euo pipefail

PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}
"$PROJECT_PYTHON" -m src.evaluation.preflight \
  --cache-root .cache/huggingface \
  --output results/preflight.json
