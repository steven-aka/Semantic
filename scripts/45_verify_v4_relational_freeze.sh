#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
from src.data.schemas import QAExample, read_jsonl
from src.representation.atomic_packet_store import AtomicPacketStore

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

config_path=Path('configs/v4_relational_precedence_ranking.json')
config=json.loads(config_path.read_text())
for path, expected in config['implementation_sha256'].items():
    assert digest(path)==expected, (path, digest(path), expected)
assert digest(config['audit']['train'])==config['audit']['train_sha256']
assert digest(config['audit']['development_confirm'])==config['audit']['development_confirm_sha256']
manifest_path=config['data']['new_candidate_manifest']
assert digest(manifest_path)==config['data']['new_candidate_manifest_sha256']
manifest=json.loads(Path(manifest_path).read_text())
assert manifest['target_outputs_observed_before_candidate_freeze'] is False
new=list(read_jsonl('data/units/qampari_rank_v4_candidates500.jsonl', QAExample))
old=list(read_jsonl('data/units/qampari_rank_v2_candidates5000.jsonl', QAExample))
assert len(new)==500 and not ({x.example_id for x in new}&{x.example_id for x in old})
store=AtomicPacketStore('data/packets_qampari_rank_v4_candidates500')
for example in new:
    packets=store.get(example.example_id)
    assert len(packets)==12
    assert '\n\n'.join(packet.text for packet in packets)==example.context.strip()
print('V4 pre-Target freeze verified: hashes, 500/500 disjoint candidates, 6000 lossless packets')
PY

source scripts/cuda_env.sh
.venv/bin/python -m unittest tests.test_weighted_precedence tests.test_v3_model_assumption_audit -q
.venv/bin/python -m compileall -q \
  src/model/relational_packet_ranker.py \
  src/model/relational_boundary_loss.py \
  src/search/weighted_precedence.py \
  src/training/train_relational_boundary_ranker.py \
  src/evaluation/relational_ranker_evaluation.py
