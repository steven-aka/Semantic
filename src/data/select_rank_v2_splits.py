from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.schemas import ExactSearchResult, QAExample, read_jsonl, write_jsonl
from src.reproducibility import sha256, tree_sha256, write_metadata


PARTITIONS = (
    ("train2000", 0, 3400, 2000),
    ("ranking_validation300", 3400, 3950, 300),
    ("calibration300", 3950, 4475, 300),
    ("final_test300", 4475, 5000, 300),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply frozen attainable-fidelity V2 split rule")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--minimum-max-fidelity", type=float, default=0.90)
    args = parser.parse_args()
    examples = list(read_jsonl(args.examples, QAExample))
    annotations = {row["example_id"]: row for row in read_jsonl(args.annotations)}
    if len(examples) != 5000:
        raise ValueError("frozen V2 primary pool requires exactly 5000 candidates")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": "v2_rank_then_cut_target_blind_partition_then_attainability_filter",
        "minimum_any_state_fidelity": args.minimum_max_fidelity,
        "source_examples_sha256": sha256(args.examples),
        "source_annotations_sha256": sha256(args.annotations),
        "partitions": {},
    }
    all_selected = set()
    for name, start, stop, required in PARTITIONS:
        eligible = []
        for example in examples[start:stop]:
            path = Path(args.exact_dir) / f"{example.example_id}.jsonl"
            exact = list(read_jsonl(path, ExactSearchResult))
            if len(exact) != 4096 or len({row.state for row in exact}) != 4096:
                raise ValueError(f"{example.example_id}: incomplete exact lattice")
            if max(row.fidelity for row in exact) >= args.minimum_max_fidelity:
                eligible.append(example)
        if len(eligible) < required:
            raise RuntimeError(
                f"{name}: only {len(eligible)} attainable candidates; require {required}; "
                "do not reallocate another partition after target outputs"
            )
        selected = eligible[:required]
        ids = {example.example_id for example in selected}
        if all_selected & ids:
            raise RuntimeError("frozen V2 partitions overlap")
        all_selected.update(ids)
        examples_path = output_dir / f"{name}.jsonl"
        annotations_path = output_dir / f"{name}_annotations.jsonl"
        write_jsonl(examples_path, selected)
        write_jsonl(annotations_path, [annotations[example.example_id] for example in selected])
        result["partitions"][name] = {
            "candidate_slice": [start, stop],
            "candidate_examples": stop - start,
            "attainable_examples": len(eligible),
            "selected_examples": len(selected),
            "selected_ids": [example.example_id for example in selected],
            "examples_sha256": sha256(examples_path),
            "annotations_sha256": sha256(annotations_path),
        }
    result["selected_id_overlap"] = 0
    result["exact_tree_sha256"] = tree_sha256(args.exact_dir)
    write_metadata(args.manifest, result)
    print(json.dumps({name: value["attainable_examples"] for name, value in result["partitions"].items()}))


if __name__ == "__main__":
    main()
