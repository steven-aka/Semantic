#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
root=results/v2_rank_then_cut
out=data/units/qampari_rank_v3_splits
mkdir -p "$out"

.venv/bin/python -m src.data.select_v3_rank_confirm \
  --examples data/units/qampari_rank_v2_candidates5000.jsonl \
  --annotations data/units/qampari_rank_v2_candidates5000_annotations.jsonl \
  --exact-dir "$root/candidates5000_exact" \
  --parent-manifest "$root/split_manifest.json" \
  --output-examples "$out/rank_confirm210.jsonl" \
  --output-annotations "$out/rank_confirm210_annotations.jsonl" \
  --manifest "$root/v3_rank_confirm_manifest.json"
