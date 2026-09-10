from __future__ import annotations

import argparse
from pathlib import Path

from src.data.schemas import ExactSearchResult, QAExample, read_jsonl, write_jsonl
from src.model.reveal_trajectory import chain_to_reveal_thresholds
from src.representation.atomic_packet_store import AtomicPacketStore
from src.search.atomic_nested_chain import best_binary_nested_chain
from src.search.exact_frontier import C_GRID
from src.reproducibility import experiment_metadata, write_metadata


def build_trajectory_row(
    example: QAExample,
    exact_rows: list[ExactSearchResult],
    packet_store: AtomicPacketStore,
    levels: tuple[float, ...] = C_GRID,
) -> dict[str, object]:
    nested = best_binary_nested_chain(exact_rows, levels)
    feasible = [row for row in nested if row["feasible"]]
    if not feasible:
        raise ValueError(f"{example.example_id}: no feasible trajectory anchors")
    states = [row["state"] for row in feasible]
    active_levels = [float(row["fidelity_level"]) for row in feasible]
    thresholds, reveal_bins, ordinal = chain_to_reveal_thresholds(
        states, active_levels
    )
    packets = packet_store.get(example.example_id)
    return {
        "example_id": example.example_id,
        "question": example.question,
        "fidelity_levels": list(levels),
        "anchor_mask": [index < len(feasible) for index in range(len(levels))],
        "packet_ids": [packet.packet_id for packet in packets],
        "packet_texts": [packet.text for packet in packets],
        "packet_tokens": [packet.tokens for packet in packets],
        "nested_states": [row["state"] if row["feasible"] else None for row in nested],
        "nested_tokens": [row["tokens"] for row in nested],
        "nested_fidelity": [row["achieved_fidelity"] for row in nested],
        "reveal_bins": reveal_bins,
        "reveal_thresholds": thresholds,
        "ordinal_labels": [
            [values[index] if index < len(feasible) else None for index in range(len(levels))]
            for values in ordinal
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert exact binary nested optima into ordinal reveal trajectories"
    )
    parser.add_argument("--examples", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output", required=True)
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
        rows.append(build_trajectory_row(example, exact, store))
    write_jsonl(args.output, rows)
    write_metadata(
        f"{args.output}.metadata.json",
        experiment_metadata(
            stage="v1_atomic_ordinal_pseudo_oracle",
            input_examples=args.examples,
            input_packets=args.packet_dir,
            input_exact=args.exact_dir,
            examples=len(rows),
            levels=list(C_GRID),
            supervision="full binary reveal trajectory with infeasible-anchor mask",
        ),
    )


if __name__ == "__main__":
    main()
