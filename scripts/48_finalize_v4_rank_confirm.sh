#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
root=results/v2_rank_then_cut
complete=$(find "$root/v4_candidates500_exact" -maxdepth 1 -name '*.jsonl' -type f | wc -l)
if [[ "$complete" != 500 ]]; then
  echo "V4 candidate exact inference incomplete: $complete/500" >&2
  exit 2
fi
mkdir -p data/units/qampari_rank_v4_splits
.venv/bin/python -m src.data.select_v4_rank_confirm \
  --examples data/units/qampari_rank_v4_candidates500.jsonl \
  --annotations data/units/qampari_rank_v4_candidates500_annotations.jsonl \
  --packet-dir data/packets_qampari_rank_v4_candidates500 \
  --exact-dir "$root/v4_candidates500_exact" \
  --candidate-manifest "$root/v4_candidates500_manifest.json" \
  --examples-output data/units/qampari_rank_v4_splits/rank_confirm300.jsonl \
  --annotations-output data/units/qampari_rank_v4_splits/rank_confirm300_annotations.jsonl \
  --manifest-output "$root/v4_rank_confirm300_manifest.json" \
  --minimum-any-state-fidelity 0.90 \
  --count 300

.venv/bin/python -m src.data.rank_oracle_dataset \
  --examples data/units/qampari_rank_v4_splits/rank_confirm300.jsonl \
  --packet-dir data/packets_qampari_rank_v4_candidates500 \
  --exact-dir "$root/v4_candidates500_exact" \
  --output "$root/v4_rank_confirm300_rank_oracle.jsonl" \
  --normalized-slack 0.005

.venv/bin/python -m src.data.critical_boundary_dataset \
  --oracle "$root/v4_rank_confirm300_rank_oracle.jsonl" \
  --exact-dir "$root/v4_candidates500_exact" \
  --output "$root/v4_rank_confirm300_boundary_oracle.jsonl" \
  --normalized-slack 0.005
