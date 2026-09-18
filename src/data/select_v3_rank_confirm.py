from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.schemas import ExactSearchResult, QAExample, read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata


def select_role_reserve(
    examples: list[QAExample],
    max_fidelity: dict[str, float],
    *,
    minimum_fidelity: float,
    consumed: int,
) -> list[QAExample]:
    eligible = [
        example
        for example in examples
        if max_fidelity[example.example_id] >= minimum_fidelity
    ]
    return eligible[consumed:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the unused V2 ranking-validation reserve for V3 confirmation"
    )
    parser.add_argument("--examples", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--output-examples", required=True)
    parser.add_argument("--output-annotations", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    all_examples = list(read_jsonl(args.examples, QAExample))
    if len(all_examples) != 5000:
        raise ValueError("V3 confirmation requires the frozen 5000-candidate pool")
    annotations = {row["example_id"]: row for row in read_jsonl(args.annotations)}
    parent = json.load(open(args.parent_manifest, encoding="utf-8"))
    partition = parent["partitions"]["ranking_validation300"]
    start, stop = partition["candidate_slice"]
    candidates = all_examples[start:stop]
    max_fidelity = {}
    for example in candidates:
        exact = list(
            read_jsonl(
                Path(args.exact_dir) / f"{example.example_id}.jsonl",
                ExactSearchResult,
            )
        )
        if len(exact) != 4096 or len({row.state for row in exact}) != 4096:
            raise ValueError(f"{example.example_id}: incomplete exact lattice")
        max_fidelity[example.example_id] = max(float(row.fidelity) for row in exact)

    minimum = float(parent["minimum_any_state_fidelity"])
    consumed = int(partition["selected_examples"])
    eligible = [
        example for example in candidates if max_fidelity[example.example_id] >= minimum
    ]
    if [row.example_id for row in eligible[:consumed]] != partition["selected_ids"]:
        raise ValueError("parent ranking-validation selection cannot be reproduced")
    selected = select_role_reserve(
        candidates,
        max_fidelity,
        minimum_fidelity=minimum,
        consumed=consumed,
    )
    if not selected:
        raise ValueError("ranking-validation partition has no unused eligible reserve")

    selected_ids = [row.example_id for row in selected]
    all_parent_ids = {
        example_id
        for role in parent["partitions"].values()
        for example_id in role["selected_ids"]
    }
    if all_parent_ids & set(selected_ids):
        raise ValueError("V3 confirmation reserve overlaps a consumed V2 role")
    write_jsonl(args.output_examples, selected)
    write_jsonl(
        args.output_annotations,
        [annotations[example_id] for example_id in selected_ids],
    )
    result = {
        "protocol": "v3_unused_v2_ranking_validation_role_reserve",
        "status": "frozen before V3 training; no calibration or final-test IDs used",
        "parent_manifest": args.parent_manifest,
        "parent_manifest_sha256": sha256(args.parent_manifest),
        "candidate_slice": [start, stop],
        "minimum_any_state_fidelity": minimum,
        "eligible_in_role": len(eligible),
        "consumed_prefix": consumed,
        "selected_examples": len(selected),
        "selected_ids": selected_ids,
        "examples": args.output_examples,
        "examples_sha256": sha256(args.output_examples),
        "annotations": args.output_annotations,
        "annotations_sha256": sha256(args.output_annotations),
        "overlap_with_all_parent_selected_roles": 0,
    }
    write_metadata(args.manifest, result)
    print(json.dumps({key: result[key] for key in ("eligible_in_role", "consumed_prefix", "selected_examples")}))


if __name__ == "__main__":
    main()
