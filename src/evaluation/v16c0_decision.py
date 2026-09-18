from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from src.data.schemas import read_jsonl
from src.reproducibility import sha256, write_metadata


def role_diagnostic(rows: list[dict]) -> dict:
    failures = [row for row in rows if row["primary_090_active"] and not row["top1_primary_090"]]
    extinct = [row for row in failures if row["first_primary_090_extinction_depth"] is not None]
    fid_rows = [row["trace"][int(row["first_primary_090_extinction_depth"]) - 1] for row in extinct]
    depths = [int(row["depth"]) for row in fid_rows]
    children = [int(row["expanded_primary_090_viable_children"]) for row in fid_rows]
    margins = [float(row["best_primary_090_margin_to_threshold"]) for row in fid_rows]
    return {
        "examples": len(rows),
        "primary_090_failures": len(failures),
        "beam_extinction_failures": len(extinct),
        "terminal_selection_failures": len(failures) - len(extinct),
        "beam_extinction_fraction_of_failures": len(extinct) / len(failures) if failures else None,
        "fid_depth_histogram": {str(key): value for key, value in sorted(Counter(depths).items())},
        "fid_depth_mean": mean(depths) if depths else None,
        "fid_depth_median": median(depths) if depths else None,
        "expanded_viable_children": {
            "minimum": min(children) if children else None,
            "median": median(children) if children else None,
            "mean": mean(children) if children else None,
            "maximum": max(children) if children else None,
        },
        "best_viable_child_margin_to_beam8_threshold": {
            "minimum": min(margins) if margins else None,
            "median": median(margins) if margins else None,
            "mean": mean(margins) if margins else None,
            "maximum": max(margins) if margins else None,
            "nonnegative": sum(value >= 0 for value in margins),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide the V16-C0 FID feasibility audit")
    parser.add_argument("--train-details", required=True)
    parser.add_argument("--validation-details", required=True)
    parser.add_argument("--train-summary", required=True)
    parser.add_argument("--validation-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    train = role_diagnostic(list(read_jsonl(args.train_details)))
    validation = role_diagnostic(list(read_jsonl(args.validation_details)))
    result = {
        "decision": "GO_V16C1_PROTOCOL_DESIGN",
        "training_authorized_by_this_audit": False,
        "reason": (
            "Primary fidelity-0.90 failure is consistently dominated by loss of every 0.90-viable "
            "beam path: 197 train-role and 19 internal-validation failures. The extinction depth and "
            "number of viable expansions align across roles, and every best viable child was below "
            "the beam-eight threshold. This supports designing a failure-triggered beam-survival "
            "objective, but its retention mix and bounded aggregation protocol must be frozen before training."
        ),
        "train2863": train,
        "internal_validation300": validation,
        "interpretation_limits": [
            "The audit establishes mechanism alignment, not that a learned correction will generalize.",
            "Exact-DP-optimal path extinction is not used as FID because it is stricter than contract viability.",
            "Complete-trajectory failures with no oracle-feasible nested trajectory are excluded from policy FID.",
            "Twenty-two failures across the two roles are terminal-selection failures and require separate handling.",
        ],
        "required_v16c1_controls": [
            "train only on train2863 failure-triggered states",
            "use primary-0.90 viable beam children as survival positives and preserve global-DP labels as an auxiliary constraint",
            "pair corrections with same-rollout pre-FID retention states",
            "freeze the replay ratio, optimizer endpoint, and at most two aggregation rounds before training",
            "evaluate first on internal-validation300; do not use development300 for iteration",
        ],
        "development300_used": False,
        "locked_roles_used": False,
        "fresh_confirmation_opened": False,
        "artifacts_sha256": {
            args.train_details: sha256(args.train_details),
            args.validation_details: sha256(args.validation_details),
            args.train_summary: sha256(args.train_summary),
            args.validation_summary: sha256(args.validation_summary),
        },
    }
    for details in (args.train_details, args.validation_details):
        compressed = f"{details}.gz"
        if Path(compressed).is_file():
            result["artifacts_sha256"][compressed] = sha256(compressed)
    write_metadata(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
