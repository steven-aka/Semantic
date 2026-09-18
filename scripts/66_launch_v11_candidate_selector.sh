#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
physical_gpu="${1:-4}"
config=configs/v11_candidate_selector.json
.venv/bin/python - "$config" <<'PY'
import hashlib,json,sys
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text())
if c['status']!='APPROVED_TO_RUN':raise SystemExit('protocol is not approved')
checks={c['data']['train_candidates']:c['data']['train_candidates_sha256'],c['data']['development_candidates']:c['data']['development_candidates_sha256'],c['data']['train_cache']:c['data']['train_cache_sha256'],c['data']['development_cache']:c['data']['development_cache_sha256'],c['frozen']['v10_head']:c['frozen']['v10_head_sha256'],**c['implementation_sha256']}
for p,h in checks.items():
 a=hashlib.sha256(Path(p).read_bytes()).hexdigest()
 if a!=h:raise SystemExit(f'hash mismatch: {p}')
print(f'preflight hashes PASS ({len(checks)} artifacts)')
PY
out=results/v2_rank_then_cut/v11_candidate_selector_seed20260918
if [[ -e "$out/candidate_selector.pt" ]]; then echo 'refusing overwrite' >&2;exit 65;fi
mkdir -p "$out";source scripts/cuda_env.sh;export CUDA_VISIBLE_DEVICES="$physical_gpu"
.venv/bin/python -u -m src.training.train_v11_candidate_selector --protocol-config "$config" --train-data results/v2_rank_then_cut/v11_train3163_candidates.jsonl --train-cache results/v2_rank_then_cut/v10_train3163_v8_embeddings.pt --development-data results/v2_rank_then_cut/v11_development300_candidates.jsonl --development-cache results/v2_rank_then_cut/v10_development300_v8_embeddings.pt --v10-head results/v2_rank_then_cut/v10_mask_value_probe_seed20260912/mask_value_head.pt --output-dir "$out" --steps 400 --batch-size 32 --lr 0.0002 --seed 20260918 2>&1 | tee "$out/run.log"
.venv/bin/python -m src.evaluation.evaluate_v11_candidate_selector --selection "$out/development_selection.jsonl" --data results/v2_rank_then_cut/v10_development300_mask_value_full.jsonl --v8-details results/v2_rank_then_cut/v8_one_run_seed20260912/checkpoints/step250/development_details.jsonl --output "$out/development_details.jsonl" | tee -a "$out/run.log"
