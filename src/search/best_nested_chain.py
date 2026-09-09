from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

from src.data.schemas import ExactSearchResult, read_jsonl
from src.search.exact_frontier import (
    C_GRID,
    attainable_levels,
    independent_frontier,
    parse_levels,
    write_csv,
)


def is_nested(lower: Sequence[int], higher: Sequence[int]) -> bool:
    return len(lower) == len(higher) and all(a <= b for a, b in zip(lower, higher))


def best_nested_chain(
    results: Sequence[ExactSearchResult], levels: Sequence[float] = C_GRID
) -> list[dict[str, object]]:
    """Minimize cumulative tokens across an ordered feasible chain using DP."""
    if not results:
        return []
    candidates = [
        sorted(
            (result for result in results if result.fidelity >= level),
            key=lambda result: (result.tokens, result.state),
        )
        for level in levels
    ]
    feasible_count = next(
        (index for index, group in enumerate(candidates) if not group), len(candidates)
    )
    if feasible_count == 0:
        return [
            {
                "example_id": results[0].example_id,
                "fidelity_level": level,
                "feasible": False,
                "tokens": None,
                "state": None,
                "achieved_fidelity": None,
                "answer_f1": None,
                "fact_recall": None,
                "cumulative_tokens": None,
            }
            for level in levels
        ]
    feasible_levels = levels[:feasible_count]
    candidates = candidates[:feasible_count]

    costs: list[list[float]] = [[float(item.tokens) for item in candidates[0]]]
    parents: list[list[int | None]] = [[None] * len(candidates[0])]
    for level_index in range(1, len(feasible_levels)):
        current_costs: list[float] = []
        current_parents: list[int | None] = []
        for current in candidates[level_index]:
            choices = [
                (costs[level_index - 1][prior_index], prior_index)
                for prior_index, prior in enumerate(candidates[level_index - 1])
                if is_nested(prior.state, current.state)
            ]
            if choices:
                prior_cost, prior_index = min(choices)
                current_costs.append(prior_cost + current.tokens)
                current_parents.append(prior_index)
            else:
                current_costs.append(math.inf)
                current_parents.append(None)
        costs.append(current_costs)
        parents.append(current_parents)

    if all(math.isinf(value) for value in costs[-1]):
        return [
            {
                "example_id": results[0].example_id,
                "fidelity_level": level,
                "feasible": False,
                "tokens": None,
                "state": None,
                "achieved_fidelity": None,
                "answer_f1": None,
                "fact_recall": None,
                "cumulative_tokens": None,
            }
            for level in levels
        ]
    index = min(range(len(costs[-1])), key=costs[-1].__getitem__)
    chosen: list[int] = [index]
    for level_index in range(len(feasible_levels) - 1, 0, -1):
        parent = parents[level_index][chosen[-1]]
        if parent is None:
            raise RuntimeError("broken nested-chain backpointer")
        chosen.append(parent)
    chosen.reverse()
    rows: list[dict[str, object]] = []
    for level_index, candidate_index in enumerate(chosen):
        result = candidates[level_index][candidate_index]
        rows.append(
            {
                "example_id": result.example_id,
                "fidelity_level": feasible_levels[level_index],
                "feasible": True,
                "tokens": result.tokens,
                "state": list(result.state),
                "achieved_fidelity": result.fidelity,
                "answer_f1": result.answer_f1,
                "fact_recall": result.fact_recall,
                "cumulative_tokens": costs[level_index][candidate_index],
            }
        )
    rows.extend(
        {
            "example_id": results[0].example_id,
            "fidelity_level": level,
            "feasible": False,
            "tokens": None,
            "state": None,
            "achieved_fidelity": None,
            "answer_f1": None,
            "fact_recall": None,
            "cumulative_tokens": None,
        }
        for level in levels[feasible_count:]
    )
    return rows


def structural_gap_rows(
    independent: Sequence[dict[str, object]], nested: Sequence[dict[str, object]]
) -> list[dict[str, object]]:
    rows = []
    for independent_row, nested_row in zip(independent, nested):
        independent_tokens = independent_row["tokens"]
        nested_tokens = nested_row["tokens"]
        feasible = independent_tokens is not None and nested_tokens is not None
        gap = int(nested_tokens) - int(independent_tokens) if feasible else None
        normalized = (
            gap / int(independent_tokens) if feasible and int(independent_tokens) > 0 else None
        )
        rows.append(
            {
                "example_id": independent_row["example_id"],
                "fidelity_level": independent_row["fidelity_level"],
                "independent_tokens": independent_tokens,
                "nested_tokens": nested_tokens,
                "structural_gap": gap,
                "structural_gap_normalized": normalized,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Find minimum-cumulative-rate nested exact chains")
    parser.add_argument("--input-dir", default="results/exact_search")
    parser.add_argument("--nested-output", default="results/nested_frontier.csv")
    parser.add_argument("--independent-output", default="results/exact_frontier.csv")
    parser.add_argument("--gap-output", default="results/structural_gap.csv")
    parser.add_argument(
        "--levels",
        default=None,
        help="strictly increasing comma-separated levels, or 'auto' for all attainable values",
    )
    args = parser.parse_args()
    paths = sorted(Path(args.input_dir).glob("*.jsonl"))
    levels = (
        C_GRID
        if args.levels is None
        else attainable_levels(paths)
        if args.levels == "auto"
        else parse_levels(args.levels)
    )
    independent_rows: list[dict[str, object]] = []
    nested_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    for path in paths:
        results = list(read_jsonl(path, ExactSearchResult))
        independent = independent_frontier(results, levels)
        nested = best_nested_chain(results, levels)
        independent_rows.extend(independent)
        nested_rows.extend(nested)
        gap_rows.extend(structural_gap_rows(independent, nested))
    write_csv(args.independent_output, independent_rows)
    write_csv(args.nested_output, nested_rows)
    write_csv(args.gap_output, gap_rows)


if __name__ == "__main__":
    main()
