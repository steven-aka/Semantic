#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/cuda_env.sh

.venv/bin/python -m unittest discover -s tests -q

.venv/bin/python - <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load(path: str):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def jsonl(path: str):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


fresh_gate = load("results/m0_qampari_atomic/fresh30_gate_result.json")
assert fresh_gate["complete"] is True
assert fresh_gate["scientific_gate_passed"] is True
assert fresh_gate["metrics"]["errors"] == []
assert all(fresh_gate["checks"].values())

config = load("configs/m0_qampari_atomic_binary_fresh30_gate.json")
assert fresh_gate["config_sha256"] == sha256(
    "configs/m0_qampari_atomic_binary_fresh30_gate.json"
)
for path, expected in config["implementation_sha256"].items():
    assert sha256(path) == expected, path

training = load("results/v1_atomic/redesign_train70/training_metadata.json")
assert training["input_data_sha256"] == sha256(training["input_data"])
assert training["best_epoch"] == 7

calibration = load("results/v1_atomic/calibration16_mapping.json")
assert calibration["mapping_nondecreasing"] is True
assert calibration["calibration_valid"] is False
assert calibration["target_calibration_success"]["0.9"] is False
assert calibration["target_calibration_success"]["0.95"] is False

fresh_summary = load("results/v1_atomic/fresh30_policy_calibrated_summary.json")
assert fresh_summary["complete"] is True
assert fresh_summary["representation_nested_fraction"] == 1.0
assert fresh_summary["calibrated"] is True
assert fresh_summary["calibration_valid"] is False
assert fresh_summary["calibration_sha256"] == sha256(
    "results/v1_atomic/calibration16_mapping.json"
)

dataset_paths = [
    "data/threshold_labels/qampari_atomic_redesign_train70_ordinal.jsonl",
    "data/threshold_labels/qampari_atomic_calibration16_ordinal.jsonl",
    "data/threshold_labels/qampari_atomic_fresh30_ordinal.jsonl",
]
id_sets = [
    {row["example_id"] for row in jsonl(path)} for path in dataset_paths
]
assert [len(ids) for ids in id_sets] == [70, 16, 30]
assert not (id_sets[0] & id_sets[1])
assert not (id_sets[0] & id_sets[2])
assert not (id_sets[1] & id_sets[2])

for directory, expected_files in (
    ("results/m0_qampari_atomic/calibration16_exact", 16),
    ("results/m0_qampari_atomic/fresh30_exact", 30),
):
    paths = sorted(Path(directory).glob("*.jsonl"))
    assert len(paths) == expected_files
    for path in paths:
        states = [tuple(row["state"]) for row in jsonl(str(path))]
        assert len(states) == 4096, path
        assert len(set(states)) == 4096, path

print("atomic M0/V1 artifact verification: PASS")
print("scientific outcome: M0 PASS; current V1 learned policy NO-GO")
PY
