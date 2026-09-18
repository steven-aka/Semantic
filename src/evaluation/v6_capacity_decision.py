from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.reproducibility import sha256, write_metadata


def development_headroom(summary: dict) -> dict:
    values = summary["per_level"]["0.9"]
    successes = int(values["contract_successes"])
    checks = {
        "fidelity_0_90_at_least_282_of_300": int(values["examples"]) == 300
        and successes >= 282,
        "all_active_trajectory_success": float(
            summary["oracle_cutoff_all_active_contracts_success_fraction"]
        )
        >= 0.90,
        "normalized_ranking_regret": float(
            summary["mean_oracle_cutoff_ranking_regret_normalized_feasible"]
        )
        <= 0.03,
    }
    passed = all(checks.values())
    return {
        "checks": checks,
        "fidelity_0_90_successes": successes,
        "passed": passed,
        "decision": (
            "GO_FREEZE_NEW_TARGET_BLIND_CONFIRM_POPULATION"
            if passed
            else "STOP_V6_WITHOUT_FRESH_CONFIRM"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select V6 capacity seed or apply its development gate")
    sub = parser.add_subparsers(dest="command", required=True)
    choose = sub.add_parser("select")
    choose.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "CHECKPOINT"), required=True)
    choose.add_argument("--config", required=True)
    choose.add_argument("--output", required=True)
    decide = sub.add_parser("decide-development")
    decide.add_argument("--summary", required=True)
    decide.add_argument("--checkpoint-metadata", required=True)
    decide.add_argument("--config", required=True)
    decide.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.command == "select":
        runs = []
        for label, checkpoint in args.run:
            metadata_path = Path(checkpoint) / "training_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            runs.append(
                {
                    "label": label,
                    "checkpoint": checkpoint,
                    "best_validation_loss": float(metadata["best_validation_loss"]),
                    "metadata_sha256": sha256(metadata_path),
                }
            )
        selected = min(runs, key=lambda row: (row["best_validation_loss"], row["label"]))
        result = {
            "complete": True,
            "selection_role": "consumed development300 objective only",
            "criterion": "minimum recorded best development tail-cost objective",
            "runs": runs,
            "selected": selected,
            "fresh_confirmation_evaluations_before_selection": 0,
        }
    else:
        summary = json.loads(Path(args.summary).read_text())
        result = development_headroom(summary)
        result.update(
            {
                "complete": True,
                "stage": "consumed_development",
                "summary": args.summary,
                "summary_sha256": sha256(args.summary),
                "checkpoint_metadata": args.checkpoint_metadata,
                "checkpoint_metadata_sha256": sha256(args.checkpoint_metadata),
            }
        )
    result.update({"config": args.config, "config_sha256": sha256(args.config)})
    write_metadata(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
