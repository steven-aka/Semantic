from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.qampari_rank_pool import build_rank_pool, certified_records
from src.data.schemas import QAExample, read_jsonl, write_jsonl
from src.representation.token_counter import load_tokenizer
from src.reproducibility import sha256, write_metadata


def select_unseen_candidate_slice(
    records: list[dict], tokenizer: object, *, start: int, count: int
) -> tuple[list[QAExample], list[dict]]:
    if start < 0 or count <= 0 or start + count > len(records):
        raise ValueError("candidate slice outside certified source pool")
    examples, annotations = build_rank_pool(records, tokenizer, n_examples=start + count)
    return examples[start : start + count], annotations[start : start + count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a new target-blind rank-confirm candidate slice")
    parser.add_argument("--input", required=True)
    parser.add_argument("--old-examples", required=True)
    parser.add_argument("--old-annotations", required=True)
    parser.add_argument("--examples-output", required=True)
    parser.add_argument("--annotations-output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--tokenizer", default="models/Qwen3-8B")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--start", type=int, default=5000)
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()
    if args.start != 5000 or args.seed != 20260910:
        raise ValueError("source-order continuation must preserve the original seed and first-5000 boundary")
    tokenizer = load_tokenizer(args.tokenizer)
    with Path(args.input).open(encoding="utf-8") as handle:
        records, rejected = certified_records(
            (json.loads(line) for line in handle if line.strip()), tokenizer, seed=args.seed
        )
    if len(records) != 6231:
        raise ValueError("certified source count changed from the frozen parent pool")
    regenerated_examples, regenerated_annotations = build_rank_pool(
        records, tokenizer, n_examples=args.start + args.count
    )
    original_examples = list(read_jsonl(args.old_examples, QAExample))
    original_annotations = list(read_jsonl(args.old_annotations))
    if original_examples != regenerated_examples[:args.start]:
        raise ValueError("canonical first 5000 examples differ from the frozen parent pool")
    if original_annotations != regenerated_annotations[:args.start]:
        raise ValueError("canonical first 5000 annotations differ from the frozen parent pool")
    examples = regenerated_examples[args.start : args.start + args.count]
    annotations = regenerated_annotations[args.start : args.start + args.count]
    old_ids = {row.example_id for row in original_examples}
    new_ids = {row.example_id for row in examples}
    if len(new_ids) != args.count or old_ids & new_ids:
        raise ValueError("new target-blind candidate IDs overlap or repeat")
    write_jsonl(args.examples_output, examples)
    write_jsonl(args.annotations_output, annotations)
    write_metadata(args.metadata_output, {
        "stage": "new_target_blind_rank_confirm_candidate_slice_before_target_inference",
        "source_split": "official_qampari_train",
        "input_sha256": sha256(args.input),
        "parent_examples_sha256": sha256(args.old_examples),
        "parent_annotations_sha256": sha256(args.old_annotations),
        "seed": args.seed,
        "certified_records": len(records),
        "candidate_slice": [args.start, args.start + args.count],
        "candidate_examples": len(examples),
        "candidate_ids": [row.example_id for row in examples],
        "overlap_with_parent_candidates": 0,
        "target_outputs_observed_before_candidate_freeze": False,
        "selection": "unchanged sha256(seed:qid) order, canonical tail after parent first 5000",
        "rejected_source_counts": rejected,
        "examples_sha256": sha256(args.examples_output),
        "annotations_sha256": sha256(args.annotations_output),
        "post_exact_eligibility": "maximum fidelity over any exact binary subset >= 0.90",
        "post_exact_confirmation_selection": "first 300 eligible candidates in frozen tail order; fail if fewer than 300",
        "locked_parent_calibration_and_final_test_used": False,
    })
    print(json.dumps({"certified": len(records), "slice": [args.start, args.start + args.count], "new": len(examples)}), flush=True)


if __name__ == "__main__":
    main()
