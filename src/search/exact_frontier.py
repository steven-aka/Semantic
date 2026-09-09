from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

from src.data.schemas import ExactSearchResult, read_jsonl


C_GRID = (0.60, 0.70, 0.80, 0.90, 0.95)


def parse_levels(value: str) -> tuple[float, ...]:
    """Parse an increasing comma-separated fidelity grid."""
    levels = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not levels:
        raise argparse.ArgumentTypeError("at least one fidelity level is required")
    if any(level <= 0.0 or level > 1.0 for level in levels):
        raise argparse.ArgumentTypeError("fidelity levels must be in (0, 1]")
    if any(lower >= higher for lower, higher in zip(levels, levels[1:])):
        raise argparse.ArgumentTypeError("fidelity levels must be strictly increasing")
    return levels


def attainable_levels(paths: Sequence[str | Path]) -> tuple[float, ...]:
    raw_levels = sorted({
        result.fidelity
        for path in paths
        for result in read_jsonl(path, ExactSearchResult)
        if result.fidelity > 0.0
    })
    if not raw_levels:
        raise ValueError("exact-search results contain no positive fidelity")
    levels: list[float] = []
    for value in raw_levels:
        if levels and math.isclose(value, levels[-1], rel_tol=1e-12, abs_tol=1e-12):
            # Use the lower representative so numerically equivalent feasible
            # states are not excluded by rounding the threshold upward.
            levels[-1] = min(levels[-1], value)
        else:
            levels.append(value)
    return tuple(levels)


def independent_optimum(
    results: Iterable[ExactSearchResult], fidelity: float
) -> ExactSearchResult | None:
    feasible = [result for result in results if result.fidelity >= fidelity]
    if not feasible:
        return None
    return min(feasible, key=lambda result: (result.tokens, sum(result.state), result.state))


def independent_frontier(
    results: Sequence[ExactSearchResult], levels: Sequence[float] = C_GRID
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    example_id = results[0].example_id if results else ""
    for level in levels:
        best = independent_optimum(results, level)
        rows.append(
            {
                "example_id": example_id,
                "fidelity_level": level,
                "feasible": best is not None,
                "tokens": best.tokens if best else None,
                "state": list(best.state) if best else None,
                "achieved_fidelity": best.fidelity if best else None,
                "answer_f1": best.answer_f1 if best else None,
                "fact_recall": best.fact_recall if best else None,
            }
        )
    return rows


def write_csv(path: str | Path, rows: Sequence[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract independent exact rate-fidelity frontiers")
    parser.add_argument("--input-dir", default="results/exact_search")
    parser.add_argument("--output", default="results/exact_frontier.csv")
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
    rows: list[dict[str, object]] = []
    for path in paths:
        rows.extend(
            independent_frontier(list(read_jsonl(path, ExactSearchResult)), levels)
        )
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
