#!/usr/bin/env bash
set -euo pipefail

PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}

"$PROJECT_PYTHON" -m src.evaluation.dry_run --output-dir results/dry_run
"$PROJECT_PYTHON" -m src.evaluation.aggregate \
  --independent results/dry_run/exact_frontier.csv \
  --nested results/dry_run/nested_frontier.csv \
  --gap results/dry_run/structural_gap.csv \
  --examples results/dry_run/examples.jsonl \
  --output results/dry_run/v0_summary.json
