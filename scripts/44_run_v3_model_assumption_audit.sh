#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
root=results/v2_rank_then_cut
for artifact in \
  "$root/v3_selected_train2000_evaluation/details.jsonl" \
  "$root/v3_selected_development300_evaluation/details.jsonl" \
  "$root/v3_rank_confirm_evaluation/details.jsonl"; do
  if [[ ! -s "$artifact" ]]; then
    echo "missing consumed-role prediction: $artifact" >&2
    exit 2
  fi
done

.venv/bin/python -m src.evaluation.v3_model_assumption_audit \
  --split train2000 "$root/train2000_rank_oracle.jsonl" "$root/v3_train2000_boundary_oracle.jsonl" \
  --split development300 "$root/ranking_validation300_rank_oracle.jsonl" "$root/v3_development300_boundary_oracle.jsonl" \
  --split confirm210 "$root/rank_confirm210_rank_oracle.jsonl" "$root/v3_rank_confirm210_boundary_oracle.jsonl" \
  --prediction train2000 "$root/v3_selected_train2000_evaluation/details.jsonl" \
  --prediction development300 "$root/v3_selected_development300_evaluation/details.jsonl" \
  --prediction confirm210 "$root/v3_rank_confirm_evaluation/details.jsonl" \
  --exact-dir "$root/candidates5000_exact" \
  --output "$root/v3_model_assumption_audit_all_consumed.json"
