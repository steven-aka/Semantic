from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_numeric(rows: Iterable[dict[str, str]], value_key: str) -> dict[float, float]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        if value not in (None, "", "None"):
            grouped[float(row["fidelity_level"])].append(float(value))
    return {level: mean(values) for level, values in sorted(grouped.items())}


def plot_v0(
    independent_path: str | Path,
    nested_path: str | Path,
    gap_path: str | Path,
    output_dir: str | Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to produce V0 figures") from exc
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    independent = aggregate_numeric(read_csv(independent_path), "tokens")
    nested = aggregate_numeric(read_csv(nested_path), "tokens")
    fig, axis = plt.subplots(figsize=(5.5, 4))
    axis.plot(independent.keys(), independent.values(), marker="o", label="Independent exact")
    axis.plot(nested.keys(), nested.values(), marker="s", label="Best nested chain")
    axis.set(xlabel="Required fidelity", ylabel="Mean target tokens")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "rate_fidelity_frontier.pdf")
    plt.close(fig)

    gaps = aggregate_numeric(read_csv(gap_path), "structural_gap_normalized")
    fig, axis = plt.subplots(figsize=(5.5, 4))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.bar(gaps.keys(), gaps.values(), width=0.035)
    axis.set(xlabel="Required fidelity", ylabel="Mean normalized structural gap")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "structural_gap.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot exact/nested V0 frontiers")
    parser.add_argument("--independent", default="results/exact_frontier.csv")
    parser.add_argument("--nested", default="results/nested_frontier.csv")
    parser.add_argument("--gap", default="results/structural_gap.csv")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()
    plot_v0(args.independent, args.nested, args.gap, args.output_dir)


if __name__ == "__main__":
    main()
