#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "V2 exact-search progress ($(date '+%F %T %Z'))"
.venv/bin/python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

examples_path = Path("data/units/qampari_rank_v2_candidates5000.jsonl")
output_dir = Path("results/v2_rank_then_cut/candidates5000_exact")
ids = []
with examples_path.open(encoding="utf-8") as handle:
    for line in handle:
        ids.append(str(json.loads(line)["example_id"]))

completed_paths = list(output_dir.glob("*.jsonl"))
completed = {path.stem for path in completed_paths}
for shard in range(6):
    expected = ids[shard::6]
    done = sum(example_id in completed for example_id in expected)
    fraction = done / len(expected) if expected else 0.0
    print(f"shard {shard}: {done:4d}/{len(expected):4d} ({fraction:6.2%})")
print(f"total:   {len(completed):4d}/{len(ids):4d} ({len(completed)/len(ids):6.2%})")
if len(completed_paths) >= 2:
    modified = [path.stat().st_mtime for path in completed_paths]
    elapsed = max(modified) - min(modified)
    if elapsed > 0:
        lattices_per_hour = (len(modified) - 1) * 3600 / elapsed
        remaining_hours = (len(ids) - len(modified)) / lattices_per_hour
        print(
            f"observed aggregate throughput: {lattices_per_hour:.2f} lattices/hour; "
            f"rough ETA: {remaining_hours:.1f} hours"
        )
        print(
            "ETA is descriptive only and is unstable while shards warm up or external GPU load changes."
        )
PY

echo
echo "Project exact-search processes"
ps -eo pid,etimes,cmd \
  | grep '[s]rc.search.atomic_exact_search' \
  || echo "none"

echo
echo "GPU index, used MiB, free MiB, utilization %, temperature C"
nvidia-smi \
  --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits

echo
echo "Refresh with: watch -n 10 bash scripts/22_monitor_rank_then_cut.sh"
