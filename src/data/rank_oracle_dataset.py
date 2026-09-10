from __future__ import annotations

import argparse
from pathlib import Path

from src.data.schemas import ExactSearchResult, QAExample, read_jsonl, write_jsonl
from src.representation.atomic_packet_store import AtomicPacketStore
from src.reproducibility import experiment_metadata, write_metadata
from src.search.near_optimal_chain_set import (
    identifiable_pair_relations,
    near_optimal_chain_membership,
    summarize_chain_membership,
)
from src.search.exact_frontier import C_GRID


def build_rank_oracle_row(
    example: QAExample,
    exact: list[ExactSearchResult],
    store: AtomicPacketStore,
    *,
    normalized_slack: float,
) -> dict[str, object]:
    membership = near_optimal_chain_membership(
        exact, C_GRID, normalized_slack=normalized_slack
    )
    packets = store.get(example.example_id)
    width = len(packets)
    relations = identifiable_pair_relations(membership, width)
    preferences = []
    for (first, second), direction in sorted(relations.items()):
        winner, loser = (first, second) if direction == -1 else (second, first)
        preferences.append([winner, loser])
    terminal = membership["state_masks_by_level"][-1]
    never_labels = []
    for packet_index in range(width):
        count = sum(bool(mask & (1 << packet_index)) for mask in terminal)
        never_labels.append(
            True if count == 0 else False if count == len(terminal) else None
        )
    summary = summarize_chain_membership(membership, width)
    return {
        "example_id": example.example_id,
        "question": example.question,
        "packet_ids": [packet.packet_id for packet in packets],
        "packet_texts": [packet.text for packet in packets],
        "packet_tokens": [packet.tokens for packet in packets],
        "active_levels": membership["active_levels"],
        "near_optimal_normalized_slack": normalized_slack,
        "optimal_cumulative_tokens": membership["optimal_cumulative_tokens"],
        "pairwise_preferences": preferences,
        "identifiable_pairs": len(preferences),
        "never_reveal_labels": never_labels,
        "packet_inclusion_fraction_by_level": [
            level["packet_inclusion_fraction"] for level in summary["levels"]
        ],
        "packet_anchor_ambiguity_fraction": summary["packet_anchor_ambiguity_fraction"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build set-valued near-optimal ranking supervision")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--normalized-slack", type=float, default=0.005)
    args = parser.parse_args()
    examples = list(read_jsonl(args.examples, QAExample))
    store = AtomicPacketStore(args.packet_dir)
    rows = []
    for example in examples:
        exact = list(
            read_jsonl(
                Path(args.exact_dir) / f"{example.example_id}.jsonl",
                ExactSearchResult,
            )
        )
        rows.append(
            build_rank_oracle_row(
                example, exact, store, normalized_slack=args.normalized_slack
            )
        )
    write_jsonl(args.output, rows)
    write_metadata(
        f"{args.output}.metadata.json",
        experiment_metadata(
            stage="v2_set_valued_near_optimal_rank_oracle",
            examples=args.examples,
            packet_dir=args.packet_dir,
            exact_dir=args.exact_dir,
            normalized_slack=args.normalized_slack,
            rows=len(rows),
            supervision="identifiable partial-order pairs; ambiguous pairs ignored",
        ),
    )


if __name__ == "__main__":
    main()
