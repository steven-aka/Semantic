"""Read-only risk/coverage diagnostic of the frozen A3-1 OOF scores."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from src.data.schemas import read_jsonl
from src.training.train_v17traj_a3_1_boundary_editor_cv import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--oof", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_READ_ONLY_DIAGNOSTIC":
        raise ValueError("protocol not frozen")
    rows = list(read_jsonl(args.oof))
    if len(rows) != 1421 or any("edit_margin" not in row for row in rows):
        raise ValueError("invalid A3-1 OOF input")
    folds = defaultdict(list)
    for index, row in enumerate(rows):
        folds[row["fold"]].append(index)
    if set(folds) != set(range(4)):
        raise ValueError("incomplete grouped folds")
    summary = {"protocol": config["protocol"], "queries": len(rows), "baseline": summarize(rows, "v8"),
               "original_a3_1": summarize(rows, "learned"), "risk_curve": [],
               "limitations": ["Fold-relative top-fraction selection is diagnostic and cannot be deployed online without a separate score calibration protocol.",
                               "All reported fractions inspect design-exposed train-only OOF outcomes; no threshold is authorized by this audit."]}
    for fraction in config["coverage_fractions"]:
        chosen = set()
        for indices in folds.values():
            eligible = [index for index in indices if rows[index]["choice"] != 0 and rows[index]["edit_margin"] > 0]
            eligible.sort(key=lambda index: (-rows[index]["edit_margin"], rows[index]["example_id"]))
            chosen.update(eligible[:math.floor(fraction * len(indices))])
        gated = []
        for index, row in enumerate(rows):
            copy = dict(row)
            if index not in chosen:
                copy["choice"] = 0
                copy["learned"] = row["v8"]
            gated.append(copy)
        summary["risk_curve"].append({"fraction": fraction, "selected_queries": len(chosen),
                                      "metrics": summarize(gated, "learned")})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
