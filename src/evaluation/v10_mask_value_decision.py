from __future__ import annotations

from typing import Any
import math


def decide_v10_probe(summary: dict[str, Any]) -> dict[str, Any]:
    errors = []
    if summary.get("complete") is not True:
        errors.append("summary is not complete")
    if int(summary.get("examples", -1)) != 300:
        errors.append("development gate requires exactly 300 examples")
    level = summary.get("per_level", {}).get("0.9", {})
    if int(level.get("examples", -1)) != 300:
        errors.append("fidelity-0.90 gate requires exactly 300 examples")
    successes = int(level.get("contract_successes", -1))
    trajectories = summary.get("oracle_cutoff_all_active_contracts_success_fraction")
    regret = summary.get("mean_oracle_cutoff_ranking_regret_normalized_feasible")
    if not isinstance(trajectories, (int, float)) or not math.isfinite(trajectories):
        errors.append("complete-trajectory fraction is missing or non-finite")
    if not isinstance(regret, (int, float)) or not math.isfinite(regret):
        errors.append("feasible normalized regret is missing or non-finite")
    if errors:
        return {"decision": "INVALID_SUMMARY", "validation_errors": errors}
    trajectories = float(trajectories)
    regret = float(regret)
    trajectory_pass = trajectories >= 0.90
    regret_pass = regret <= 0.03
    if successes >= 282 and trajectory_pass and regret_pass:
        decision = "GO_FREEZE_SAME_HEAD_FOR_NEW_CONFIRMATION"
    elif 279 <= successes <= 281 and trajectory_pass and regret_pass:
        decision = "INCONCLUSIVE_NO_FRESH_ROLE"
    else:
        decision = "STOP_MASK_VALUE_HYPOTHESIS"
    return {
        "decision": decision,
        "fidelity_0.90_successes": successes,
        "fidelity_0.90_gate": successes >= 282,
        "complete_trajectory_fraction": trajectories,
        "complete_trajectory_gate": trajectory_pass,
        "feasible_normalized_regret": regret,
        "regret_gate": regret_pass,
    }
