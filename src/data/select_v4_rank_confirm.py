from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.data.schemas import ExactSearchResult, QAExample, read_jsonl, write_jsonl
from src.representation.atomic_packet_store import AtomicPacketStore
from src.reproducibility import sha256, write_metadata
from src.search.atomic_exact_search import validate_atomic_exact_rows


def tree_digest(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate V4 exact tail and freeze first 300 eligible confirmation examples")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--examples-output", required=True)
    parser.add_argument("--annotations-output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--minimum-any-state-fidelity", type=float, default=0.90)
    parser.add_argument("--count", type=int, default=300)
    args = parser.parse_args()
    candidates = list(read_jsonl(args.examples, QAExample))
    annotations = {row["example_id"]: row for row in read_jsonl(args.annotations)}
    if len(candidates) != 500 or len(annotations) != 500:
        raise ValueError("frozen V4 candidate tail must contain exactly 500 examples")
    store = AtomicPacketStore(args.packet_dir)
    exact_dir = Path(args.exact_dir)
    eligible = []
    exact_files = []
    for index, example in enumerate(candidates, 1):
        path = exact_dir / f"{example.example_id}.jsonl"
        if not path.is_file():
            raise ValueError(f"missing exact lattice: {example.example_id}")
        packets = store.get(example.example_id)
        rows = list(read_jsonl(path, ExactSearchResult))
        errors = validate_atomic_exact_rows(example, packets, rows)
        if errors:
            raise ValueError(f"{example.example_id}: {'; '.join(errors)}")
        exact_files.append(path)
        if max(float(row.fidelity) for row in rows) >= args.minimum_any_state_fidelity:
            eligible.append(example)
        if index % 50 == 0:
            print(json.dumps({"validated": index, "total": len(candidates), "eligible": len(eligible)}), flush=True)
    unexpected = sorted(
        path.name for path in exact_dir.glob("*.jsonl")
        if path.stem not in {example.example_id for example in candidates}
    )
    if unexpected:
        raise ValueError("exact directory contains examples outside the frozen candidate tail")
    if len(eligible) < args.count:
        raise ValueError(f"only {len(eligible)} eligible examples; protocol requires {args.count} without replacement")
    selected = eligible[: args.count]
    selected_annotations = [annotations[example.example_id] for example in selected]
    write_jsonl(args.examples_output, selected)
    write_jsonl(args.annotations_output, selected_annotations)
    write_metadata(args.manifest_output, {
        "stage": "v4_one_shot_rank_confirmation_frozen_after_complete_exact_validation",
        "candidate_manifest": args.candidate_manifest,
        "candidate_manifest_sha256": sha256(args.candidate_manifest),
        "candidate_examples_sha256": sha256(args.examples),
        "candidate_annotations_sha256": sha256(args.annotations),
        "exact_tree_sha256": tree_digest(sorted(exact_files)),
        "candidate_examples": len(candidates),
        "eligible_examples": len(eligible),
        "minimum_any_state_fidelity": args.minimum_any_state_fidelity,
        "selection": "first 300 eligible in the pre-Target frozen candidate-tail order",
        "selected_examples": len(selected),
        "selected_ids": [example.example_id for example in selected],
        "examples_sha256": sha256(args.examples_output),
        "annotations_sha256": sha256(args.annotations_output),
        "overlap_with_parent_candidate5000": 0,
        "model_evaluations_before_selection": 0,
        "calibration300_or_final_test300_used": False,
    })
    print(json.dumps({"complete": True, "eligible": len(eligible), "selected": len(selected)}), flush=True)


if __name__ == "__main__":
    main()
