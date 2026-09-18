#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
root=results/v2_rank_then_cut
mkdir -p "$root/v3_boundary_diagnostic"

.venv/bin/python -m src.evaluation.critical_boundary_diagnostic \
  --oracle "$root/ranking_validation300_rank_oracle.jsonl" \
  --exact-dir "$root/candidates5000_exact" \
  --run ranknet_20260910 "$root/evaluation/train2000_seed20260910_details.jsonl" \
  --run ranknet_20260911 "$root/evaluation/train2000_seed20260911_details.jsonl" \
  --run ranknet_20260912 "$root/evaluation/train2000_seed20260912_details.jsonl" \
  --run listwise_20260910 "$root/listwise_evaluation/train2000_seed20260910_details.jsonl" \
  --run listwise_20260911 "$root/listwise_evaluation/train2000_seed20260911_details.jsonl" \
  --run listwise_20260912 "$root/listwise_evaluation/train2000_seed20260912_details.jsonl" \
  --normalized-slack 0.005 \
  --familywise-alpha 0.05 \
  --required-contract-lower-bound 0.90 \
  --details "$root/v3_boundary_diagnostic/details.jsonl" \
  --output "$root/v3_boundary_diagnostic/summary.json"
