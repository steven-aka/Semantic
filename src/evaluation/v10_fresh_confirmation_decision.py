from __future__ import annotations

from typing import Any

from src.evaluation.v4_relational_decision import fresh_confirmation


def decide_v10_fresh_confirmation(summary: dict[str, Any]) -> dict[str, Any]:
    errors = []
    if summary.get("complete") is not True:
        errors.append("summary is not complete")
    if int(summary.get("examples", -1)) != 300:
        errors.append("fresh confirmation requires exactly 300 examples")
    if set(summary.get("per_level", {})) != {"0.6", "0.7", "0.8", "0.9", "0.95"}:
        errors.append("fresh confirmation requires all five fidelity levels")
    if errors:
        return {"decision": "INVALID_SUMMARY", "validation_errors": errors}
    result = fresh_confirmation(summary, familywise_alpha=0.05)
    result["decision"] = (
        "GO_CUTOFF_AND_CALIBRATION" if result["passed"] else "STOP_MASK_VALUE_AFTER_FRESH_CONFIRM"
    )
    return result
