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
log_root = root / "logs"
processes = subprocess.run(
    ["ps", "-ww", "-eo", "pid,etimes,args"],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()

complete = running = interrupted = 0
for size in (500, 1000, 2000):
    for seed in (20260910, 20260911, 20260912):
        name = f"train{size}_seed{seed}"
        directory = root / name
        log_path = log_root / f"{name}.log"
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
            epochs = []
            if log_path.exists():
                for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith('{"epoch"'):
                        try:
                            epochs.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            progress = ""
            if epochs:
                latest = epochs[-1]
                progress = (
                    f" epoch={latest['epoch']}/5 "
                    f"validation_loss={latest['validation_example_mean_pairwise_loss']:.6f}"
                )
            print(
                f"{name}: RUNNING pid={fields[0]} elapsed={elapsed / 3600:.2f}h"
                f"{progress}"
            )
        elif log_path.exists():
            interrupted += 1
            print(f"{name}: INTERRUPTED log={log_path}")
        else:
            print(f"{name}: PENDING")
print(
    f"summary: complete={complete}/9 running={running} "
    f"interrupted={interrupted} pending={9-complete-running-interrupted}"
)
PY

echo
echo "GPU index, used MiB, free MiB, utilization %, temperature C"
nvidia-smi \
  --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits
echo "Physical GPU4 is excluded from this project by operator instruction."

echo
echo "Refresh with: watch -n 10 bash scripts/25_monitor_v2_rank_training.sh"
