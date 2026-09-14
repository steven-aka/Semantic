#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "V2 rank-only training status ($(date '+%F %T %Z'))"
.venv/bin/python - <<'PY'
from __future__ import annotations

import json
import subprocess
from pathlib import Path

root = Path("results/v2_rank_then_cut/training")
processes = subprocess.run(
    ["ps", "-eo", "pid,etimes,cmd"],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()

complete = running = 0
for size in (500, 1000, 2000):
    for seed in (20260910, 20260911, 20260912):
        name = f"train{size}_seed{seed}"
        directory = root / name
        metadata = directory / "training_metadata.json"
        matches = [line.strip() for line in processes if str(directory) in line]
        if metadata.exists():
            row = json.loads(metadata.read_text(encoding="utf-8"))
            complete += 1
            print(
                f"{name}: COMPLETE best_epoch={row['best_epoch']} "
                f"validation_loss={row['best_validation_loss']:.6f} "
                f"elapsed={row['elapsed_seconds'] / 3600:.2f}h"
            )
        elif matches:
            running += 1
            fields = matches[0].split(maxsplit=2)
            elapsed = int(fields[1])
            print(f"{name}: RUNNING pid={fields[0]} elapsed={elapsed / 3600:.2f}h")
        else:
            print(f"{name}: PENDING")
print(f"summary: complete={complete}/9 running={running} pending={9-complete-running}")
PY

echo
echo "GPU index, used MiB, free MiB, utilization %, temperature C"
nvidia-smi \
  --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits
echo "Physical GPU4 is excluded from this project by operator instruction."

echo
echo "Refresh with: watch -n 10 bash scripts/25_monitor_v2_rank_training.sh"
