#!/usr/bin/env bash
set -euo pipefail

PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}
export MPLCONFIGDIR=${MPLCONFIGDIR:-"$PWD/.cache/matplotlib"}
mkdir -p "$MPLCONFIGDIR"

"$PROJECT_PYTHON" -m src.search.best_nested_chain \
  --input-dir results/exact_search \
  --independent-output results/exact_frontier.csv \
  --nested-output results/nested_frontier.csv \
  --gap-output results/structural_gap.csv
"$PROJECT_PYTHON" -m src.evaluation.frontier_metrics \
  --independent results/exact_frontier.csv \
  --nested results/nested_frontier.csv \
  --gap results/structural_gap.csv \
  --output-dir results/figures
"$PROJECT_PYTHON" -m src.evaluation.aggregate \
  --independent results/exact_frontier.csv \
  --nested results/nested_frontier.csv \
  --gap results/structural_gap.csv \
  --examples data/units/hotpot_controlled_pilot_eligible.jsonl \
  --output results/v0_summary.json
