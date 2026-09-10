from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.schemas import ExactSearchResult, read_jsonl
from src.reproducibility import sha256, write_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate target fidelity to policy control fidelity")
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-success", type=float, default=0.90)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 < args.minimum_success <= 1 or not 0 < args.step <= 0.25:
        parser.error("invalid calibration success or step")

    rows = list(read_jsonl(args.data))
    predictions = {
        row["example_id"]: row for row in read_jsonl(args.predictions)
    }
    exact_by_example = {
        row["example_id"]: {
            result.state: result
            for result in read_jsonl(
                Path(args.exact_dir) / f"{row['example_id']}.jsonl",
                ExactSearchResult,
            )
        }
        for row in rows
    }
    levels = [float(value) for value in rows[0]["fidelity_levels"]]
    chosen = []
    target_success = []
    diagnostics = []
    lower_bound = 0.0
    for anchor_index, target in enumerate(levels):
        candidates = []
        value = target
        while value < 1.0 - 1e-9:
            candidates.append(round(value, 10))
            value += args.step
        candidates.append(1.0)
        candidate_rows = []
        selected = None
        for control in candidates:
            hits = count = 0
            for row in rows:
                if not row["anchor_mask"][anchor_index]:
                    continue
                threshold = predictions[row["example_id"]]["predicted_thresholds"]
                state = tuple(int(control >= value) for value in threshold)
                hits += int(exact_by_example[row["example_id"]][state].fidelity >= target)
                count += 1
            fraction = hits / count if count else 0.0
            candidate_rows.append(
                {"control_level": control, "successes": hits, "examples": count, "success_fraction": fraction}
            )
            if selected is None and control >= lower_bound and fraction >= args.minimum_success:
                selected = control
        meets_minimum = selected is not None
        if selected is None:
            selected = 1.0
        selected = max(selected, lower_bound)
        lower_bound = selected
        chosen.append(selected)
        target_success.append(meets_minimum)
        diagnostics.append(
            {
                "target_level": target,
                "selected_control_level": selected,
                "meets_minimum_empirical_success": meets_minimum,
                "candidates": candidate_rows,
            }
        )

    result = {
        "protocol": "v1_atomic_monotone_fidelity_calibration",
        "minimum_empirical_success": args.minimum_success,
        "candidate_step": args.step,
        "target_levels": levels,
        "control_levels": {str(target): control for target, control in zip(levels, chosen)},
        "mapping_nondecreasing": all(a <= b for a, b in zip(chosen, chosen[1:])),
        "target_calibration_success": {
            str(target): success for target, success in zip(levels, target_success)
        },
        "calibration_valid": all(target_success),
        "fallback_control_1_0_is_not_success": True,
        "diagnostics": diagnostics,
        "artifacts": {
            "data_sha256": sha256(args.data),
            "predictions_sha256": sha256(args.predictions),
            "exact_tree": args.exact_dir,
        },
    }
    write_metadata(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
