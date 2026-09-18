from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.evaluation.critical_boundary_diagnostic import (
    derive_critical_boundary_preferences,
)
from src.reproducibility import experiment_metadata, sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.search.near_optimal_chain_set import near_optimal_chain_membership


def build_boundary_row(
    source: dict[str, object],
    exact: list[ExactSearchResult],
    *,
    normalized_slack: float,
) -> dict[str, object]:
    width = len(source["packet_ids"])
    levels = [float(value) for value in source["active_levels"]]
    membership = near_optimal_chain_membership(
        exact, levels, normalized_slack=normalized_slack
    )
    fidelity_by_mask = {state_to_mask(row.state): float(row.fidelity) for row in exact}
    boundaries = derive_critical_boundary_preferences(
        fidelity_by_mask,
        membership["state_masks_by_level"],
        membership["active_levels"],
        width,
        source["pairwise_preferences"],
    )
    output = dict(source)
    output["all_identifiable_preferences"] = output["pairwise_preferences"]
    output["all_identifiable_pairs"] = output["identifiable_pairs"]
    output["pairwise_preferences"] = [
        list(pair) for pair in boundaries["stable_counterfactual"]
    ]
    output["identifiable_pairs"] = len(output["pairwise_preferences"])
    output["boundary_supervision"] = {
        "definition": "local add-drop critical-harmful witness intersected with set-identifiable near-optimal-chain precedence",
        "local_counterfactual_directed_pairs": boundaries[
            "local_counterfactual_directed_pairs"
        ],
        "conflicting_local_unordered_pairs": boundaries[
            "conflicting_local_unordered_pairs"
        ],
        "critical_packet_witnesses": boundaries["critical_packet_witnesses"],
        "harmful_packet_witnesses": boundaries["harmful_packet_witnesses"],
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build exhaustive set-valued stable critical-boundary supervision"
    )
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--normalized-slack", type=float, default=0.005)
    args = parser.parse_args()

    source_rows = list(read_jsonl(args.oracle))
    rows = []
    for source in source_rows:
        exact = list(
            read_jsonl(
                Path(args.exact_dir) / f"{source['example_id']}.jsonl",
                ExactSearchResult,
            )
        )
        rows.append(
            build_boundary_row(
                source, exact, normalized_slack=args.normalized_slack
            )
        )
    write_jsonl(args.output, rows)
    pair_counts = [int(row["identifiable_pairs"]) for row in rows]
    conflicts = [
        int(row["boundary_supervision"]["conflicting_local_unordered_pairs"])
        for row in rows
    ]
    metadata = experiment_metadata(
        stage="v3_set_valued_stable_critical_boundary_oracle",
        source_oracle=args.oracle,
        source_oracle_sha256=sha256(args.oracle),
        exact_dir=args.exact_dir,
        normalized_slack=args.normalized_slack,
        examples=len(rows),
        supervised_examples=sum(count > 0 for count in pair_counts),
        zero_boundary_examples=sum(count == 0 for count in pair_counts),
        boundary_pairs=sum(pair_counts),
        mean_boundary_pairs=mean(pair_counts),
        examples_with_raw_local_direction_conflicts=sum(count > 0 for count in conflicts),
        filtering="retain only local critical-harmful directions identifiable across the complete near-optimal chain set",
    )
    write_metadata(f"{args.output}.metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
