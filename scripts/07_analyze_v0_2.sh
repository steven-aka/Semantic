#!/usr/bin/env bash
set -euo pipefail

PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}
export MPLCONFIGDIR=${MPLCONFIGDIR:-"$PWD/.cache/matplotlib"}
mkdir -p "$MPLCONFIGDIR" results/v0_2/figures results/v0_2/figures_attainable

"$PROJECT_PYTHON" -m src.search.best_nested_chain \
  --input-dir results/v0_2/exact_search \
  --independent-output results/v0_2/exact_frontier.csv \
  --nested-output results/v0_2/nested_frontier.csv \
  --gap-output results/v0_2/structural_gap.csv
"$PROJECT_PYTHON" -m src.evaluation.frontier_metrics \
  --independent results/v0_2/exact_frontier.csv \
  --nested results/v0_2/nested_frontier.csv \
  --gap results/v0_2/structural_gap.csv \
  --output-dir results/v0_2/figures
"$PROJECT_PYTHON" -m src.evaluation.aggregate \
  --independent results/v0_2/exact_frontier.csv \
  --nested results/v0_2/nested_frontier.csv \
  --gap results/v0_2/structural_gap.csv \
  --examples data/units/hotpot_fact_atomic_eligible.jsonl \
  --output results/v0_2/v0_summary.json
"$PROJECT_PYTHON" -m src.evaluation.frontier_diagnostics \
  --independent results/v0_2/exact_frontier.csv \
  --gap results/v0_2/structural_gap.csv \
  --output results/v0_2/diagnostics.json

"$PROJECT_PYTHON" -m src.search.best_nested_chain \
  --input-dir results/v0_2/exact_search \
  --levels auto \
  --independent-output results/v0_2/exact_frontier_attainable.csv \
  --nested-output results/v0_2/nested_frontier_attainable.csv \
  --gap-output results/v0_2/structural_gap_attainable.csv
"$PROJECT_PYTHON" -m src.evaluation.frontier_metrics \
  --independent results/v0_2/exact_frontier_attainable.csv \
  --nested results/v0_2/nested_frontier_attainable.csv \
  --gap results/v0_2/structural_gap_attainable.csv \
  --output-dir results/v0_2/figures_attainable
"$PROJECT_PYTHON" -m src.evaluation.frontier_diagnostics \
  --independent results/v0_2/exact_frontier_attainable.csv \
  --gap results/v0_2/structural_gap_attainable.csv \
  --output results/v0_2/diagnostics_attainable.json
