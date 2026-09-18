#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
output="results/v2_rank_then_cut/v10_mask_value_probe_seed20260912"

echo "== protocol =="
.venv/bin/python - <<'PY'
import json
from pathlib import Path
c = json.loads(Path("configs/v10_mask_value_probe.json").read_text())
print("status:", c["status"])
print("output:", c["output"])
PY

echo "== process =="
ps -eo pid,etimes,cmd | grep '[t]rain_v10_mask_value_probe' || true

echo "== GPUs =="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

echo "== latest log =="
if [[ -f "$output/run.log" ]]; then
  tail -n 30 "$output/run.log"
else
  echo "no run.log (training has not started)"
fi

echo "== result =="
if [[ -f "$output/development_summary.json" ]]; then
  .venv/bin/python - <<'PY'
import json
from pathlib import Path
from src.evaluation.v10_mask_value_decision import decide_v10_probe
p = Path("results/v2_rank_then_cut/v10_mask_value_probe_seed20260912/development_summary.json")
s = json.loads(p.read_text())
print(json.dumps(decide_v10_probe(s), indent=2))
PY
else
  echo "no development_summary.json"
fi
