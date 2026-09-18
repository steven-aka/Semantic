from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.rank_failure_audit import clopper_pearson_lower, minimum_successes_for_lower_bound
from src.reproducibility import sha256, write_metadata


def development_headroom(summary: dict) -> dict:
    values = summary["per_level"]["0.9"]
    successes = int(
        values["contract_successes"]
        if "contract_successes" in values
        else round(values["contract_success_fraction"] * values["examples"])
    )
    checks = {
        "fidelity_0_90_at_least_282_of_300": int(values["examples"]) == 300 and successes >= 282,
        "all_active_trajectory_success": float(summary["oracle_cutoff_all_active_contracts_success_fraction"]) >= 0.90,
        "normalized_ranking_regret": float(summary["mean_oracle_cutoff_ranking_regret_normalized_feasible"]) <= 0.03,
    }
    return {
        "checks": checks,
        "fidelity_0_90_successes": successes,
        "passed": all(checks.values()),
        "decision": "GO_ONE_SHOT_FRESH_CONFIRM" if all(checks.values()) else "STOP_V4_WITHOUT_FRESH_CONFIRM",
    }


def fresh_confirmation(summary: dict, *, familywise_alpha: float = 0.05) -> dict:
    levels = sorted(summary["per_level"], key=float)
    corrected = familywise_alpha / len(levels)
    risk = {}
    for level in levels:
        values = summary["per_level"][level]
        examples = int(values["examples"])
        successes = int(
            values["contract_successes"]
            if "contract_successes" in values
            else round(values["contract_success_fraction"] * examples)
        )
        required = minimum_successes_for_lower_bound(examples, 0.90, corrected)
        lower = clopper_pearson_lower(successes, examples, corrected)
        risk[level] = {
            "examples": examples, "successes": successes,
            "required_successes": required,
            "bonferroni_clopper_pearson_lower": lower,
            "passed": lower >= 0.90,
        }
    checks = {
        "all_per_anchor_risk_bounds": all(row["passed"] for row in risk.values()),
        "active_contract_success": float(summary["oracle_cutoff_active_contract_success_fraction"]) >= 0.97,
        "all_active_trajectory_success": float(summary["oracle_cutoff_all_active_contracts_success_fraction"]) >= 0.90,
        "normalized_ranking_regret": float(summary["mean_oracle_cutoff_ranking_regret_normalized_feasible"]) <= 0.03,
    }
    return {
        "checks": checks, "per_anchor_risk": risk,
        "passed": all(checks.values()),
        "decision": "GO_PREREGISTER_CUTOFF" if all(checks.values()) else "STOP_V4_RANKING",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select V4 seed or apply its frozen development/fresh gate")
    sub = parser.add_subparsers(dest="command", required=True)
    choose = sub.add_parser("select")
    choose.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "CHECKPOINT"), required=True)
    choose.add_argument("--config", required=True)
    choose.add_argument("--output", required=True)
    decide = sub.add_parser("decide")
    decide.add_argument("--stage", choices=("development", "fresh"), required=True)
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
            runs.append({
                "label": label, "checkpoint": checkpoint,
                "best_validation_loss": float(metadata["best_validation_loss"]),
                "metadata_sha256": sha256(metadata_path),
            })
        selected = min(runs, key=lambda row: (row["best_validation_loss"], row["label"]))
        result = {
            "complete": True, "selection_role": "consumed development300 only",
            "criterion": "minimum recorded best development worst-relational-boundary loss",
            "runs": runs, "selected": selected,
            "config": args.config, "config_sha256": sha256(args.config),
            "fresh_confirmation_evaluations_before_selection": 0,
        }
    else:
        summary = json.loads(Path(args.summary).read_text())
        result = development_headroom(summary) if args.stage == "development" else fresh_confirmation(summary)
        result.update({
            "complete": True, "stage": args.stage,
            "summary": args.summary, "summary_sha256": sha256(args.summary),
            "checkpoint_metadata": args.checkpoint_metadata,
            "checkpoint_metadata_sha256": sha256(args.checkpoint_metadata),
            "config": args.config, "config_sha256": sha256(args.config),
        })
    write_metadata(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
