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

amendment2_path = Path("configs/v2_rank_then_cut_pretraining_amendment2.json")
expected_amendment2_sha = "ed84f603809f82a1da68b57d3c7153914a26564ef230ee10828114b1dfc2ef7c"
assert sha256(amendment2_path) == expected_amendment2_sha, "pre-training amendment 2 changed"
amendment2 = json.loads(amendment2_path.read_text(encoding="utf-8"))
assert amendment2["parent_protocol_sha256"] == expected_config_sha
assert amendment2["prior_amendment_sha256"] == sha256(amendment_path)
superseded = set(amendment2["implementation_sha256"])
for addition in amendment["additions"].values():
    if addition["path"] not in superseded:
        assert sha256(addition["path"]) == addition["sha256"], addition["path"]

for path, expected in config["implementation_sha256"].items():
    if path not in superseded:
        assert sha256(path) == expected, f"implementation changed: {path}"
for path, expected in amendment2["implementation_sha256"].items():
    assert sha256(path) == expected, f"amendment 2 implementation changed: {path}"
for key in ("split_manifest", "train2000_oracle", "ranking_validation300_oracle"):
    path = amendment2["trigger"][key]
    assert sha256(path) == amendment2["trigger"][f"{key}_sha256"], path

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
