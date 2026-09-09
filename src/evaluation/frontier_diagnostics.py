from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.search.best_nested_chain import is_nested


def _rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def diagnose_frontier(
    independent_rows: list[dict[str, str]], gap_rows: list[dict[str, str]]
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in independent_rows:
        if row["state"] not in ("", "None"):
            grouped[row["example_id"]].append(row)

    adjacent_pairs = state_switches = rate_increases = fact_recall_increases = 0
    nonnested_switches = examples_with_switch = 0
    for rows in grouped.values():
        rows.sort(key=lambda row: float(row["fidelity_level"]))
        switched = False
        for lower, higher in zip(rows, rows[1:]):
            adjacent_pairs += 1
            low_state = list(ast.literal_eval(lower["state"]))
            high_state = list(ast.literal_eval(higher["state"]))
            if low_state != high_state:
                switched = True
                state_switches += 1
                nonnested_switches += int(not is_nested(low_state, high_state))
            rate_increases += int(float(higher["tokens"]) > float(lower["tokens"]))
            if lower.get("fact_recall") not in (None, "", "None") and higher.get(
                "fact_recall"
            ) not in (None, "", "None"):
                fact_recall_increases += int(
                    float(higher["fact_recall"]) > float(lower["fact_recall"])
                )
        examples_with_switch += int(switched)

    comparable_gaps = [
        float(row["structural_gap"])
        for row in gap_rows
        if row["structural_gap"] not in ("", "None")
    ]
    if state_switches == 0 or rate_increases == 0:
        assessment = "no_go_representation_or_fidelity_method_not_identifiable"
        next_action = (
            "stop_before_v1_and_revise_representation_or_fidelity_measurement"
        )
    elif nonnested_switches == 0:
        assessment = "tradeoff_observed_but_structural_gap_not_stress_tested"
        next_action = (
            "do_not_interpret_zero_gap_as_evidence; revise_the_bounded_v0_pilot_design"
        )
    else:
        assessment = "structural_gap_identifiable"
        next_action = "perform_human_v0_gate_review"
    return {
        "n_examples": len(grouped),
        "adjacent_feasible_pairs": adjacent_pairs,
        "state_switches": state_switches,
        "state_switch_rate": state_switches / adjacent_pairs if adjacent_pairs else None,
        "examples_with_any_state_switch": examples_with_switch,
        "strict_rate_increases": rate_increases,
        "fact_recall_increases": fact_recall_increases,
        "nonnested_independent_switches": nonnested_switches,
        "positive_structural_gap_rows": sum(value > 0 for value in comparable_gaps),
        "assessment": assessment,
        "next_action": next_action,
        "bounded_decision_rule": (
            "Stop before V1 when the rate-fidelity frontier is flat. If switches exist "
            "but none is nonnested, keep the zero-gap result descriptive and revise one "
            "bounded V0 pilot design rather than adding representation levels."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether V0 can identify refinement cost")
    parser.add_argument("--independent", required=True)
    parser.add_argument("--gap", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = diagnose_frontier(_rows(args.independent), _rows(args.gap))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
