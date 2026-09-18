from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from src.data.schemas import read_jsonl, write_jsonl
from src.reproducibility import experiment_metadata, sha256, write_metadata


def _edge_record(
    row: dict[str, Any], winner: int, loser: int, *, safety: bool
) -> dict[str, Any]:
    fractions = row["packet_inclusion_fraction_by_level"]
    gaps = [
        max(0.0, float(level[winner]) - float(level[loser]))
        for level in fractions
    ]
    total_packet_tokens = sum(int(value) for value in row["packet_tokens"])
    if total_packet_tokens <= 0:
        raise ValueError(f"{row['example_id']}: packet token total must be positive")
    dominance_mass = sum(gaps)
    anchor_coverage = sum(gap > 0.0 for gap in gaps)
    loser_token_fraction = int(row["packet_tokens"][loser]) / total_packet_tokens
    rate_exposure = dominance_mass * loser_token_fraction
    raw_weight = (
        1.0 + dominance_mass + rate_exposure
        if safety
        else dominance_mass * loser_token_fraction
    )
    return {
        "winner": winner,
        "loser": loser,
        "raw_weight": raw_weight,
        "anchor_coverage": anchor_coverage,
        "dominance_mass": dominance_mass,
        "loser_packet_tokens": int(row["packet_tokens"][loser]),
        "loser_token_fraction": loser_token_fraction,
        "rate_exposure": rate_exposure,
    }


def _normalize(records: list[dict[str, Any]]) -> None:
    if not records:
        return
    scale = mean(float(record["raw_weight"]) for record in records)
    if scale <= 0:
        raise ValueError("edge weights must have positive mean")
    for record in records:
        record["weight"] = float(record["raw_weight"]) / scale


def build_tail_cost_row(source: dict[str, Any]) -> dict[str, Any]:
    safety_pairs = {
        (int(pair[0]), int(pair[1])) for pair in source["pairwise_preferences"]
    }
    all_pairs = {
        (int(pair[0]), int(pair[1]))
        for pair in source["all_identifiable_preferences"]
    }
    if not safety_pairs <= all_pairs:
        raise ValueError(f"{source['example_id']}: safety edges must be identifiable")
    safety = [
        _edge_record(source, winner, loser, safety=True)
        for winner, loser in sorted(safety_pairs)
    ]
    rate = [
        _edge_record(source, winner, loser, safety=False)
        for winner, loser in sorted(all_pairs - safety_pairs)
    ]
    # An identifiable direction must have some earlier-anchor inclusion gap.
    rate = [record for record in rate if float(record["raw_weight"]) > 0]
    _normalize(safety)
    _normalize(rate)
    output = dict(source)
    output["safety_preferences"] = safety
    output["rate_preferences"] = rate
    output["tail_cost_supervision"] = {
        "safety_edges": len(safety),
        "rate_edges": len(rate),
        "safety_weight": "within-example mean-normalized 1 + sum_anchor inclusion_gap + loser_rate_exposure",
        "rate_weight": "within-example mean-normalized sum_anchor inclusion_gap * loser_packet_token_fraction",
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V5 tail-risk and cost-aware edge supervision")
    parser.add_argument("--boundary-oracle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source_rows = list(read_jsonl(args.boundary_oracle))
    rows = [build_tail_cost_row(row) for row in source_rows]
    write_jsonl(args.output, rows)
    safety_counts = [len(row["safety_preferences"]) for row in rows]
    rate_counts = [len(row["rate_preferences"]) for row in rows]
    metadata = experiment_metadata(
        stage="v5_tail_risk_cost_aware_supervision",
        source_boundary_oracle=args.boundary_oracle,
        source_boundary_oracle_sha256=sha256(args.boundary_oracle),
        examples=len(rows),
        safety_edges=sum(safety_counts),
        rate_edges=sum(rate_counts),
        examples_with_safety=sum(count > 0 for count in safety_counts),
        examples_with_rate=sum(count > 0 for count in rate_counts),
        safety_weight="within-example mean-normalized 1 + dominance_mass + rate_exposure",
        rate_weight="within-example mean-normalized dominance_mass * loser packet token fraction",
        dominance_mass="sum over active anchors of max(0, inclusion_fraction_winner - inclusion_fraction_loser)",
    )
    write_metadata(f"{args.output}.metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
