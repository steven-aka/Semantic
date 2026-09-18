from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.rank_failure_audit import (
    clopper_pearson_lower,
    minimum_successes_for_lower_bound,
)
from src.reproducibility import sha256, write_metadata


def decide_boundary_confirmation(
    summary: dict[str, object],
    *,
    familywise_alpha: float,
    required_lower_bound: float,
    minimum_active_success: float,
    minimum_trajectory_success: float,
    maximum_regret: float,
) -> dict[str, object]:
    levels = sorted(summary["per_level"], key=float)
    corrected_alpha = familywise_alpha / len(levels)
    risk = {}
    for level in levels:
        values = summary["per_level"][level]
        examples = int(values["examples"])
        successes = round(float(values["contract_success_fraction"]) * examples)
        required = minimum_successes_for_lower_bound(
            examples, required_lower_bound, corrected_alpha
        )
        lower = clopper_pearson_lower(successes, examples, corrected_alpha)
        risk[level] = {
            "examples": examples,
            "successes": successes,
            "required_successes": required,
            "bonferroni_clopper_pearson_lower": lower,
            "passed": lower >= required_lower_bound,
        }
    checks = {
        "all_per_anchor_risk_bounds": all(row["passed"] for row in risk.values()),
        "active_contract_success": float(
            summary["oracle_cutoff_active_contract_success_fraction"]
        )
        >= minimum_active_success,
        "all_active_trajectory_success": float(
            summary["oracle_cutoff_all_active_contracts_success_fraction"]
        )
        >= minimum_trajectory_success,
        "normalized_ranking_regret": float(
            summary["mean_oracle_cutoff_ranking_regret_normalized_feasible"]
        )
        <= maximum_regret,
    }
    return {
        "checks": checks,
        "per_anchor_risk": risk,
        "scientific_gate_passed": all(checks.values()),
        "decision": "GO_IMPLEMENT_CUTOFF" if all(checks.values()) else "STOP_V3_RANKING",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide one-shot V3 rank confirmation")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--checkpoint-metadata", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = json.load(open(args.summary, encoding="utf-8"))
    config = json.load(open(args.config, encoding="utf-8"))
    gate = config["one_shot_rank_confirmation_gate"]
    result = decide_boundary_confirmation(
        summary,
        familywise_alpha=float(gate["familywise_alpha"]),
        required_lower_bound=float(gate["minimum_per_anchor_contract_lower_bound"]),
        minimum_active_success=float(gate["minimum_active_contract_success"]),
        minimum_trajectory_success=float(gate["minimum_all_active_trajectory_success"]),
        maximum_regret=float(gate["maximum_normalized_ranking_regret_feasible"]),
    )
    result.update(
        {
            "complete": True,
            "status": "one_shot_fresh_rank_confirm_before_cutoff",
            "summary": args.summary,
            "summary_sha256": sha256(args.summary),
            "checkpoint_metadata": args.checkpoint_metadata,
            "checkpoint_metadata_sha256": sha256(args.checkpoint_metadata),
            "config": args.config,
            "config_sha256": sha256(args.config),
        }
    )
    write_metadata(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
