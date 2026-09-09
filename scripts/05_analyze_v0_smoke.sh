#!/usr/bin/env bash
set -euo pipefail

PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}
export MPLCONFIGDIR=${MPLCONFIGDIR:-"$PWD/.cache/matplotlib"}
mkdir -p "$MPLCONFIGDIR" results/smoke/figures

"$PROJECT_PYTHON" -m src.search.best_nested_chain \
  --input-dir results/exact_search_smoke \
  --independent-output results/smoke/exact_frontier.csv \
  --nested-output results/smoke/nested_frontier.csv \
  --gap-output results/smoke/structural_gap.csv
"$PROJECT_PYTHON" -m src.evaluation.frontier_metrics \
  --independent results/smoke/exact_frontier.csv \
  --nested results/smoke/nested_frontier.csv \
  --gap results/smoke/structural_gap.csv \
  --output-dir results/smoke/figures
"$PROJECT_PYTHON" -m src.evaluation.aggregate \
  --independent results/smoke/exact_frontier.csv \
  --nested results/smoke/nested_frontier.csv \
  --gap results/smoke/structural_gap.csv \
  --examples data/units/hotpot_exact_pilot_eligible.jsonl \
  --output results/smoke/v0_summary.json
