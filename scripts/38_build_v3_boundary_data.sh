#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
root=results/v2_rank_then_cut
exact="$root/candidates5000_exact"
confirm_examples=data/units/qampari_rank_v3_splits/rank_confirm210.jsonl

.venv/bin/python -m src.data.rank_oracle_dataset \
  --examples "$confirm_examples" \
  --packet-dir data/packets_qampari_rank_v2_candidates5000 \
  --exact-dir "$exact" \
  --output "$root/rank_confirm210_rank_oracle.jsonl" \
  --normalized-slack 0.005

.venv/bin/python -m src.data.critical_boundary_dataset \
  --oracle "$root/rank_learning_curve/train2000_rank_oracle.jsonl" \
  --exact-dir "$exact" \
  --output "$root/v3_train2000_boundary_oracle.jsonl" \
  --normalized-slack 0.005

.venv/bin/python -m src.data.critical_boundary_dataset \
  --oracle "$root/ranking_validation300_rank_oracle.jsonl" \
  --exact-dir "$exact" \
  --output "$root/v3_development300_boundary_oracle.jsonl" \
  --normalized-slack 0.005

.venv/bin/python -m src.data.critical_boundary_dataset \
  --oracle "$root/rank_confirm210_rank_oracle.jsonl" \
  --exact-dir "$exact" \
  --output "$root/v3_rank_confirm210_boundary_oracle.jsonl" \
  --normalized-slack 0.005
