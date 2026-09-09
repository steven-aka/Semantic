#!/usr/bin/env bash
set -euo pipefail

PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}
export MPLCONFIGDIR=${MPLCONFIGDIR:-"$PWD/.cache/matplotlib"}
RESULT_DIR=${RESULT_DIR:-results/v0_4}
mkdir -p "$MPLCONFIGDIR" "$RESULT_DIR/figures" "$RESULT_DIR/figures_attainable"

"$PROJECT_PYTHON" -m src.search.best_nested_chain \
  --input-dir "$RESULT_DIR/exact_search" \
  --independent-output "$RESULT_DIR/exact_frontier.csv" \
  --nested-output "$RESULT_DIR/nested_frontier.csv" \
  --gap-output "$RESULT_DIR/structural_gap.csv"
"$PROJECT_PYTHON" -m src.evaluation.frontier_metrics \
  --independent "$RESULT_DIR/exact_frontier.csv" --nested "$RESULT_DIR/nested_frontier.csv" \
  --gap "$RESULT_DIR/structural_gap.csv" --output-dir "$RESULT_DIR/figures"
"$PROJECT_PYTHON" -m src.evaluation.aggregate \
  --independent "$RESULT_DIR/exact_frontier.csv" --nested "$RESULT_DIR/nested_frontier.csv" \
  --gap "$RESULT_DIR/structural_gap.csv" --output "$RESULT_DIR/v0_summary.json"
"$PROJECT_PYTHON" -m src.evaluation.frontier_diagnostics \
  --independent "$RESULT_DIR/exact_frontier.csv" --gap "$RESULT_DIR/structural_gap.csv" \
  --output "$RESULT_DIR/diagnostics.json"

"$PROJECT_PYTHON" -m src.search.best_nested_chain \
  --input-dir "$RESULT_DIR/exact_search" --levels auto \
  --independent-output "$RESULT_DIR/exact_frontier_attainable.csv" \
  --nested-output "$RESULT_DIR/nested_frontier_attainable.csv" \
  --gap-output "$RESULT_DIR/structural_gap_attainable.csv"
"$PROJECT_PYTHON" -m src.evaluation.frontier_metrics \
  --independent "$RESULT_DIR/exact_frontier_attainable.csv" \
  --nested "$RESULT_DIR/nested_frontier_attainable.csv" \
  --gap "$RESULT_DIR/structural_gap_attainable.csv" --output-dir "$RESULT_DIR/figures_attainable"
"$PROJECT_PYTHON" -m src.evaluation.frontier_diagnostics \
  --independent "$RESULT_DIR/exact_frontier_attainable.csv" \
  --gap "$RESULT_DIR/structural_gap_attainable.csv" \
  --output "$RESULT_DIR/diagnostics_attainable.json"
