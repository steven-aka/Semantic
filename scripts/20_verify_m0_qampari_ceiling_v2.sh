#!/usr/bin/env bash
set -euo pipefail
. scripts/cuda_env.sh

PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}

"$PROJECT_PYTHON" -m compileall -q src tests
"$PROJECT_PYTHON" -m unittest discover -s tests -v

"$PROJECT_PYTHON" -m src.evaluation.qampari_gate \
  --config configs/m0_qampari_ceiling_v2_dev_gate.json \
  --examples data/units/qampari_ceiling_v2_dev30_highstate.jsonl \
  --annotations data/units/qampari_ceiling_v2_dev30_highstate_annotations.jsonl \
  --selection-manifest results/m0_qampari_ceiling_v2/dev30_highstate_selection_manifest.json \
  --packet-dir data/packets_qampari_ceiling_v2_dev30_highstate_lossless \
  --exact-dir results/m0_qampari_ceiling_v2/dev30_exact_search \
  --ceiling-baseline results/m0_qampari_ceiling_v2/high_state_candidates98.jsonl \
  --output-dir results/m0_qampari_ceiling_v2/dev \
  --output results/m0_qampari_ceiling_v2/dev_gate_result.json

"$PROJECT_PYTHON" -m src.evaluation.qampari_gate \
  --config configs/m0_qampari_ceiling_v2_heldout_gate.json \
  --examples data/units/qampari_ceiling_v2_heldout30.jsonl \
  --annotations data/units/qampari_ceiling_v2_heldout30_annotations.jsonl \
  --selection-manifest results/m0_qampari_ceiling_v2/heldout30_selection_manifest.json \
  --packet-dir data/packets_qampari_ceiling_v2_heldout30_lossless \
  --exact-dir results/m0_qampari_ceiling_v2/heldout30_exact_search \
  --ceiling-baseline results/m0_qampari_ceiling_v2/high_state_candidates98.jsonl \
  --output-dir results/m0_qampari_ceiling_v2/heldout \
  --output results/m0_qampari_ceiling_v2/heldout_gate_result.json

"$PROJECT_PYTHON" - <<'PY'
import json
from pathlib import Path


def read_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_ids(path: str):
    return {
        json.loads(line)["example_id"]
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


dev = read_json("results/m0_qampari_ceiling_v2/dev_gate_result.json")
heldout = read_json("results/m0_qampari_ceiling_v2/heldout_gate_result.json")
legacy = read_json("results/m0_qampari/lossless_dev_gate_result.json")
dev_ids = read_ids("data/units/qampari_ceiling_v2_dev30_highstate.jsonl")
heldout_ids = read_ids("data/units/qampari_ceiling_v2_heldout30.jsonl")

assert dev["complete"] and dev["scientific_gate_passed"] and not dev["metrics"]["errors"]
assert heldout["complete"] and heldout["scientific_gate_passed"] and not heldout["metrics"]["errors"]
assert dev["metrics"]["exact_states"] == heldout["metrics"]["exact_states"] == 30 * 729
assert not dev_ids.intersection(heldout_ids)
assert legacy["complete"] and not legacy["scientific_gate_passed"]
assert legacy["metrics"]["max_f1_0_9_examples"] == 25
print("M0 ceiling-v2 verification passed: dev GO, disjoint heldout GO, legacy NO-GO preserved.")
PY

echo "Verification complete. V1 readiness review is permitted; V1 training is not started by this script."
