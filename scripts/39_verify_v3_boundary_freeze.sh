#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/cuda_env.sh
bash scripts/31_verify_v2_1_listwise_freeze.sh

.venv/bin/python - <<'PY'
import json
from pathlib import Path

from src.data.schemas import read_jsonl
from src.reproducibility import sha256

path = Path('configs/v3_critical_boundary_ranking.json')
expected = '2828df23db28771b2c6c97abefbc75da547b0944becf6852253d04dbba12e829'
assert sha256(path) == expected, 'V3 critical-boundary protocol changed'
config = json.load(open(path))
assert sha256(config['parent_protocol']['path']) == config['parent_protocol']['sha256']
for key in ('v2_1_decision', 'boundary_diagnostic'):
    assert sha256(config['trigger'][key]) == config['trigger'][f'{key}_sha256']
for key in ('train', 'consumed_development', 'rank_confirm_manifest', 'rank_confirm'):
    assert sha256(config['data'][key]) == config['data'][f'{key}_sha256']
for implementation, digest in config['implementation_sha256'].items():
    assert sha256(implementation) == digest, implementation

train_ids = {row['example_id'] for row in read_jsonl(config['data']['train'])}
dev_ids = {
    row['example_id'] for row in read_jsonl(config['data']['consumed_development'])
}
confirm_ids = {row['example_id'] for row in read_jsonl(config['data']['rank_confirm'])}
assert len(train_ids) == 2000
assert len(dev_ids) == 300
assert len(confirm_ids) == 210
assert not train_ids & dev_ids
assert not train_ids & confirm_ids
assert not dev_ids & confirm_ids

decision = json.load(open(config['trigger']['v2_1_decision']))
assert decision['decision'] == 'STOP_V2_1_RANKING'
diagnostic = json.load(open(config['trigger']['boundary_diagnostic']))
assert diagnostic['complete']
for run in diagnostic['runs'].values():
    assert run['stable_counterfactual_projection']['per_level']['0.9']['successes'] == 298
assert not Path('results/v2_rank_then_cut/v3_rank_confirm_evaluation/summary.json').exists()
assert not Path('results/v2_rank_then_cut/v3_rank_confirm_decision.json').exists()
assert not Path('results/v2_rank_then_cut/calibration').exists()
assert not Path('results/v2_rank_then_cut/final_test').exists()
print('V3 critical-boundary pre-training freeze: PASS')
PY
