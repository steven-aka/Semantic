#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/cuda_env.sh

.venv/bin/python - <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.reproducibility import sha256, tree_sha256

config_path = Path("configs/v2_rank_then_cut_hypothesis.json")
expected_config_sha = "2376fd274372ac6c595e1cfeb30a13c36dc23d72c302fcec308f89563e7aeb79"
assert sha256(config_path) == expected_config_sha, "frozen protocol config changed"
config = json.loads(config_path.read_text(encoding="utf-8"))

amendment_path = Path("configs/v2_rank_then_cut_pretraining_amendment1.json")
amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
assert amendment["parent_protocol_sha256"] == expected_config_sha
for addition in amendment["additions"].values():
    assert sha256(addition["path"]) == addition["sha256"], addition["path"]

for path, expected in config["implementation_sha256"].items():
    assert sha256(path) == expected, f"implementation changed: {path}"

pool = config["candidate_pool"]
assert sha256(pool["examples"]) == pool["examples_sha256"]
assert sha256(pool["annotations"]) == pool["annotations_sha256"]
assert tree_sha256(pool["packet_dir"]) == pool["packet_tree_sha256"]

example_count = sum(1 for line in Path(pool["examples"]).open(encoding="utf-8") if line.strip())
annotation_count = sum(1 for line in Path(pool["annotations"]).open(encoding="utf-8") if line.strip())
packet_count = len(list(Path(pool["packet_dir"]).glob("*.jsonl")))
assert (example_count, annotation_count, packet_count) == (5000, 5000, 5000)

exact_dir = Path("results/v2_rank_then_cut/candidates5000_exact")
complete = len(list(exact_dir.glob("*.jsonl")))
print("V2 frozen protocol/data verification: PASS")
print(f"exact progress: {complete}/5000 complete example lattices")
PY

.venv/bin/python -m unittest discover -s tests -q
echo "V2 implementation tests: PASS"
