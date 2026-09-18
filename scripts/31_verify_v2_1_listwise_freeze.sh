#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/cuda_env.sh

bash scripts/23_verify_rank_then_cut_freeze.sh

.venv/bin/python - <<'PY'
import json
from pathlib import Path

from src.reproducibility import sha256

path = Path("configs/v2_1_listwise_ranking_amendment.json")
expected = "8f55715ab5adacb74de2fa748992db37b2e8813e02f48f2edf4450dc0300a4d0"
assert sha256(path) == expected, "V2.1 listwise amendment changed"
config = json.loads(path.read_text(encoding="utf-8"))
assert sha256(config["parent_protocol"]) == config["parent_protocol_sha256"]
for key in ("rank_only_gate", "rank_failure_audit", "projection_diagnostic"):
    assert sha256(config["trigger"][key]) == config["trigger"][f"{key}_sha256"]
for key in ("train_data", "validation_data"):
    assert sha256(config["preserved"][key]) == config["preserved"][f"{key}_sha256"]
for implementation, digest in config["implementation_sha256"].items():
    assert sha256(implementation) == digest, implementation
assert not Path("results/v2_rank_then_cut/calibration").exists()
assert not Path("results/v2_rank_then_cut/final_test").exists()
print("V2.1 exact partial-order listwise freeze: PASS")
PY
