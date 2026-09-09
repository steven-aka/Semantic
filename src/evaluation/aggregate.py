from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from src.data.schemas import QAExample, read_jsonl
from src.search.best_nested_chain import is_nested


def _csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_v0_summary(
    independent_rows: list[dict[str, str]],
    nested_rows: list[dict[str, str]],
    gap_rows: list[dict[str, str]],
    examples: Iterable[QAExample] = (),
) -> dict[str, Any]:
    example_map = {example.example_id: example for example in examples}
    levels = sorted({float(row["fidelity_level"]) for row in independent_rows})
    by_level: dict[str, dict[str, Any]] = {}
    for level in levels:
        key = f"{level:.2f}"
        independent = [
            float(row["tokens"])
            for row in independent_rows
            if float(row["fidelity_level"]) == level and row["tokens"] not in ("", "None")
        ]
        nested = [
            float(row["tokens"])
            for row in nested_rows
            if float(row["fidelity_level"]) == level and row["tokens"] not in ("", "None")
        ]
        normalized_gaps = [
            float(row["structural_gap_normalized"])
            for row in gap_rows
            if float(row["fidelity_level"]) == level
            and row["structural_gap_normalized"] not in ("", "None")
        ]
        by_level[key] = {
            "independent_feasible_examples": len(independent),
            "nested_feasible_examples": len(nested),
            "mean_independent_tokens": mean(independent) if independent else None,
            "mean_nested_tokens": mean(nested) if nested else None,
            "mean_structural_gap_normalized": mean(normalized_gaps) if normalized_gaps else None,
        }

    nested_by_example: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in nested_rows:
        nested_by_example[row["example_id"]].append(row)
    adjacency_count = 0
    nestedness_violations = 0
    rate_decreases = 0
    support_visibility: dict[float, list[float]] = defaultdict(list)
    for example_id, rows in nested_by_example.items():
        rows.sort(key=lambda row: float(row["fidelity_level"]))
        parsed: list[tuple[float, list[int], int]] = []
        for row in rows:
            if row["state"] in ("", "None"):
                continue
            parsed.append(
                (
                    float(row["fidelity_level"]),
                    list(ast.literal_eval(row["state"])),
                    int(float(row["tokens"])),
                )
            )
        for previous, current in zip(parsed, parsed[1:]):
            adjacency_count += 1
            nestedness_violations += int(not is_nested(previous[1], current[1]))
            rate_decreases += int(current[2] < previous[2])
        example = example_map.get(example_id)
        if example:
            supporting = [index for index, unit in enumerate(example.units) if unit.supporting]
            for level, state, _tokens in parsed:
                visibility = (
                    sum(state[index] >= 1 for index in supporting) / len(supporting)
                    if supporting
                    else 1.0
                )
                support_visibility[level].append(visibility)
    return {
        "label": "V0 exact-pilot summary",
        "n_examples": len(nested_by_example),
        "by_fidelity_level": by_level,
        "representation_nestedness": {
            "adjacent_pairs": adjacency_count,
            "violations": nestedness_violations,
            "rate_decreases": rate_decreases,
        },
        "mean_supporting_unit_visibility": {
            f"{level:.2f}": mean(values) for level, values in sorted(support_visibility.items())
        },
        "gate_status": "requires_human_review_of_real_results",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate descriptive V0 gate metrics")
    parser.add_argument("--independent", default="results/exact_frontier.csv")
    parser.add_argument("--nested", default="results/nested_frontier.csv")
    parser.add_argument("--gap", default="results/structural_gap.csv")
    parser.add_argument("--examples", default="data/units/hotpot_exact_pilot_eligible.jsonl")
    parser.add_argument("--output", default="results/v0_summary.json")
    args = parser.parse_args()
    examples = list(read_jsonl(args.examples, QAExample)) if Path(args.examples).exists() else []
    summary = build_v0_summary(_csv(args.independent), _csv(args.nested), _csv(args.gap), examples)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()

