#!/usr/bin/env bash
set -euo pipefail

PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}
export MPLCONFIGDIR=${MPLCONFIGDIR:-"$PWD/.cache/matplotlib"}
mkdir -p "$MPLCONFIGDIR" results/v0_1

"$PROJECT_PYTHON" -m src.evaluation.rescore_fact_fidelity \
  --input-dir results/exact_search \
  --coverage results/v0_1/fact_coverage.jsonl \
  --output-dir results/v0_1/exact_search
"$PROJECT_PYTHON" -m src.search.best_nested_chain \
  --input-dir results/v0_1/exact_search \
  --independent-output results/v0_1/exact_frontier.csv \
  --nested-output results/v0_1/nested_frontier.csv \
  --gap-output results/v0_1/structural_gap.csv
"$PROJECT_PYTHON" -m src.evaluation.frontier_metrics \
  --independent results/v0_1/exact_frontier.csv \
  --nested results/v0_1/nested_frontier.csv \
  --gap results/v0_1/structural_gap.csv \
  --output-dir results/v0_1/figures
"$PROJECT_PYTHON" -m src.evaluation.aggregate \
  --independent results/v0_1/exact_frontier.csv \
  --nested results/v0_1/nested_frontier.csv \
  --gap results/v0_1/structural_gap.csv \
  --examples data/units/hotpot_controlled_pilot_eligible.jsonl \
  --output results/v0_1/v0_summary.json
"$PROJECT_PYTHON" -m src.evaluation.frontier_diagnostics \
  --independent results/v0_1/exact_frontier.csv \
  --gap results/v0_1/structural_gap.csv \
  --output results/v0_1/diagnostics.json

# The preregistered 0.60--0.95 grid remains the primary result above.  This
# secondary grid includes the attainable fact-recall breakpoints for examples
# with only 2--5 gold facts, so a flat primary frontier can be distinguished
# from a genuinely constant representation.
SENSITIVITY_LEVELS=${SENSITIVITY_LEVELS:-auto}
"$PROJECT_PYTHON" -m src.search.best_nested_chain \
  --input-dir results/v0_1/exact_search \
  --levels "$SENSITIVITY_LEVELS" \
  --independent-output results/v0_1/exact_frontier_attainable_grid.csv \
  --nested-output results/v0_1/nested_frontier_attainable_grid.csv \
  --gap-output results/v0_1/structural_gap_attainable_grid.csv
"$PROJECT_PYTHON" -m src.evaluation.frontier_diagnostics \
  --independent results/v0_1/exact_frontier_attainable_grid.csv \
  --gap results/v0_1/structural_gap_attainable_grid.csv \
  --output results/v0_1/diagnostics_attainable_grid.json
