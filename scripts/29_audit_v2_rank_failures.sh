#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/python -m src.evaluation.rank_failure_audit \
  --oracle results/v2_rank_then_cut/ranking_validation300_rank_oracle.jsonl \
  --exact-dir results/v2_rank_then_cut/candidates5000_exact \
  --run 20260910 results/v2_rank_then_cut/evaluation/train2000_seed20260910_details.jsonl \
  --run 20260911 results/v2_rank_then_cut/evaluation/train2000_seed20260911_details.jsonl \
  --run 20260912 results/v2_rank_then_cut/evaluation/train2000_seed20260912_details.jsonl \
  --details-output results/v2_rank_then_cut/rank_failure_audit_details.jsonl \
  --summary-output results/v2_rank_then_cut/rank_failure_audit_summary.json
