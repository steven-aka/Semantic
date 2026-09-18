from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from src.data.critical_boundary_dataset import build_boundary_row
from src.data.rank_oracle_dataset import build_rank_oracle_row
from src.data.schemas import ExactSearchResult, QAExample, read_jsonl, write_jsonl
from src.data.tail_cost_boundary_dataset import build_tail_cost_row
from src.representation.atomic_packet_store import AtomicPacketStore
from src.reproducibility import experiment_metadata, sha256, write_metadata


def select_all_attainable(
    candidates: list[QAExample],
    max_fidelity: dict[str, float],
    parent_selected_ids: list[str],
    *,
    minimum_fidelity: float,
) -> list[QAExample]:
    selected = [
        example
        for example in candidates
        if max_fidelity[example.example_id] >= minimum_fidelity
    ]
    selected_ids = [example.example_id for example in selected]
    if selected_ids[: len(parent_selected_ids)] != parent_selected_ids:
        raise ValueError("expanded train no longer reproduces the frozen train2000 prefix")
    return selected


def _derive_new_row(
    item: tuple[QAExample, str, str, float, float]
) -> tuple[str, float, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    example, exact_dir, packet_dir, minimum_fidelity, normalized_slack = item
    exact = list(
        read_jsonl(
            Path(exact_dir) / f"{example.example_id}.jsonl",
            ExactSearchResult,
        )
    )
    if len(exact) != 4096 or len({row.state for row in exact}) != 4096:
        raise ValueError(f"{example.example_id}: incomplete exact lattice")
    maximum = max(float(row.fidelity) for row in exact)
    if maximum < minimum_fidelity:
        return example.example_id, maximum, None, None, None
    rank = build_rank_oracle_row(
        example,
        exact,
        AtomicPacketStore(packet_dir),
        normalized_slack=normalized_slack,
    )
    boundary = build_boundary_row(rank, exact, normalized_slack=normalized_slack)
    tail_cost = build_tail_cost_row(boundary)
    return example.example_id, maximum, rank, boundary, tail_cost


def _ordered_rows(ids: Iterable[str], rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [rows[example_id] for example_id in ids]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the deterministic all-attainable V7 training-role expansion"
    )
    parser.add_argument("--examples", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--base-rank", required=True)
    parser.add_argument("--base-boundary", required=True)
    parser.add_argument("--base-tail-cost", required=True)
    parser.add_argument("--output-examples", required=True)
    parser.add_argument("--output-rank", required=True)
    parser.add_argument("--output-boundary", required=True)
    parser.add_argument("--output-tail-cost", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--normalized-slack", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    all_examples = list(read_jsonl(args.examples, QAExample))
    if len(all_examples) != 5000:
        raise ValueError("V7 requires the frozen 5000-candidate parent pool")
    parent = json.load(open(args.parent_manifest, encoding="utf-8"))
    role = parent["partitions"]["train2000"]
    start, stop = (int(value) for value in role["candidate_slice"])
    candidates = all_examples[start:stop]
    parent_ids = [str(value) for value in role["selected_ids"]]
    parent_set = set(parent_ids)
    minimum = float(parent["minimum_any_state_fidelity"])

    base_paths = {
        "rank": args.base_rank,
        "boundary": args.base_boundary,
        "tail_cost": args.base_tail_cost,
    }
    base_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for role_name, path in base_paths.items():
        rows = list(read_jsonl(path))
        ids = [str(row["example_id"]) for row in rows]
        if ids != parent_ids:
            raise ValueError(f"{role_name} base rows do not match frozen train2000 order")
        base_rows[role_name] = {str(row["example_id"]): row for row in rows}

    unseen = [example for example in candidates if example.example_id not in parent_set]
    tasks = [
        (example, args.exact_dir, args.packet_dir, minimum, args.normalized_slack)
        for example in unseen
    ]
    derived: dict[str, tuple[float, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(_derive_new_row, tasks, chunksize=1), 1):
            example_id, maximum, rank, boundary, tail_cost = result
            derived[example_id] = (maximum, rank, boundary, tail_cost)
            if index % 100 == 0:
                print(f"processed unseen candidates: {index}/{len(tasks)}", flush=True)

    max_fidelity = {example_id: minimum for example_id in parent_ids}
    max_fidelity.update({example_id: value[0] for example_id, value in derived.items()})
    selected = select_all_attainable(
        candidates,
        max_fidelity,
        parent_ids,
        minimum_fidelity=minimum,
    )
    selected_ids = [example.example_id for example in selected]
    expected = int(role["attainable_examples"])
    if len(selected) != expected:
        raise ValueError(f"expected {expected} attainable train examples, found {len(selected)}")

    expanded: dict[str, dict[str, dict[str, Any]]] = {
        name: dict(rows) for name, rows in base_rows.items()
    }
    for example_id in selected_ids[len(parent_ids) :]:
        _, rank, boundary, tail_cost = derived[example_id]
        if rank is None or boundary is None or tail_cost is None:
            raise AssertionError(f"{example_id}: selected row lacks derived supervision")
        expanded["rank"][example_id] = rank
        expanded["boundary"][example_id] = boundary
        expanded["tail_cost"][example_id] = tail_cost

    write_jsonl(args.output_examples, selected)
    write_jsonl(args.output_rank, _ordered_rows(selected_ids, expanded["rank"]))
    write_jsonl(args.output_boundary, _ordered_rows(selected_ids, expanded["boundary"]))
    write_jsonl(args.output_tail_cost, _ordered_rows(selected_ids, expanded["tail_cost"]))

    tail_rows = _ordered_rows(selected_ids, expanded["tail_cost"])
    group_counts = Counter(example.dataset for example in selected)
    manifest = experiment_metadata(
        stage="v7_all_attainable_frozen_training_role_expansion",
        status="frozen after V6 stopped; before V7 training; no fresh, calibration, or final-test access",
        parent_manifest=args.parent_manifest,
        parent_manifest_sha256=sha256(args.parent_manifest),
        candidate_slice=[start, stop],
        candidate_examples=len(candidates),
        minimum_any_state_fidelity=minimum,
        original_train_examples=len(parent_ids),
        added_train_examples=len(selected_ids) - len(parent_ids),
        selected_examples=len(selected_ids),
        original_train_is_exact_prefix=True,
        selection="all attainable examples in the pre-existing frozen train candidate slice, preserving source order",
        dataset_counts=dict(sorted(group_counts.items())),
        safety_edges=sum(len(row["safety_preferences"]) for row in tail_rows),
        rate_edges=sum(len(row["rate_preferences"]) for row in tail_rows),
        output_examples=args.output_examples,
        output_examples_sha256=sha256(args.output_examples),
        output_rank=args.output_rank,
        output_rank_sha256=sha256(args.output_rank),
        output_boundary=args.output_boundary,
        output_boundary_sha256=sha256(args.output_boundary),
        output_tail_cost=args.output_tail_cost,
        output_tail_cost_sha256=sha256(args.output_tail_cost),
        base_artifacts={name: {"path": path, "sha256": sha256(path)} for name, path in base_paths.items()},
        selected_ids=selected_ids,
        calibration_or_final_test_used=False,
        fresh_confirmation_used=False,
    )
    write_metadata(args.manifest, manifest)
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "selected_examples",
                    "added_train_examples",
                    "dataset_counts",
                    "safety_edges",
                    "rate_edges",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
