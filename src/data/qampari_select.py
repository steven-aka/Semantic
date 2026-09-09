from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data.schemas import QAExample, read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata


def select_qampari_development(
    examples: Sequence[QAExample],
    annotations: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    n_examples: int = 30,
    minimum_full_f1: float = 0.8,
    maximum_empty_f1: float = 0.2,
    minimum_context_gain: float = 0.6,
    require_nontruncated: bool = False,
    eligible_offset: int = 0,
) -> tuple[list[QAExample], list[Mapping[str, Any]]]:
    if eligible_offset < 0:
        raise ValueError("eligible_offset must be non-negative")
    annotations_by_id = {str(row["example_id"]): row for row in annotations}
    baseline_by_id = {str(row["example_id"]): row for row in baseline_rows}
    eligible_examples = []
    eligible_annotations = []
    for example in examples:
        if example.example_id not in annotations_by_id or example.example_id not in baseline_by_id:
            raise ValueError(f"missing QAMPARI development input for {example.example_id}")
        row = baseline_by_id[example.example_id]
        if (
            float(row["full_metrics"]["f1"]) >= minimum_full_f1
            and float(row["empty_metrics"]["f1"]) <= maximum_empty_f1
            and float(row["f1_context_gain"]) >= minimum_context_gain
            and (
                not require_nontruncated
                or str(row.get("full_finish_reason", "unknown")) != "length"
            )
        ):
            eligible_examples.append(example)
            eligible_annotations.append(annotations_by_id[example.example_id])
        if len(eligible_examples) == eligible_offset + n_examples:
            break
    selected_examples = eligible_examples[eligible_offset:eligible_offset + n_examples]
    selected_annotations = eligible_annotations[eligible_offset:eligible_offset + n_examples]
    if len(selected_examples) != n_examples:
        raise ValueError(
            f"only {len(eligible_examples)} examples pass through requested offset; "
            f"require {eligible_offset + n_examples}"
        )
    return selected_examples, selected_annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze answerable QAMPARI M0 development examples")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--examples-output", required=True)
    parser.add_argument("--annotations-output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--n-examples", type=int, default=30)
    parser.add_argument("--minimum-full-f1", type=float, default=0.8)
    parser.add_argument("--maximum-empty-f1", type=float, default=0.2)
    parser.add_argument("--minimum-context-gain", type=float, default=0.6)
    parser.add_argument("--require-nontruncated", action="store_true")
    parser.add_argument("--eligible-offset", type=int, default=0)
    args = parser.parse_args()
    examples = list(read_jsonl(args.examples, QAExample))
    annotations = list(read_jsonl(args.annotations))
    baseline = list(read_jsonl(args.baseline))
    selected_examples, selected_annotations = select_qampari_development(
        examples,
        annotations,
        baseline,
        n_examples=args.n_examples,
        minimum_full_f1=args.minimum_full_f1,
        maximum_empty_f1=args.maximum_empty_f1,
        minimum_context_gain=args.minimum_context_gain,
        require_nontruncated=args.require_nontruncated,
        eligible_offset=args.eligible_offset,
    )
    write_jsonl(args.examples_output, selected_examples)
    write_jsonl(args.annotations_output, selected_annotations)
    write_metadata(
        args.manifest_output,
        {
            "stage": "m0_qampari_development_selection_frozen_before_packets",
            "scientific_status": "development_only_not_a_locked_result",
            "source_examples_sha256": sha256(args.examples),
            "source_annotations_sha256": sha256(args.annotations),
            "source_baseline_sha256": sha256(args.baseline),
            "n_examples": args.n_examples,
            "minimum_full_f1": args.minimum_full_f1,
            "maximum_empty_f1": args.maximum_empty_f1,
            "minimum_context_gain": args.minimum_context_gain,
            "require_nontruncated": args.require_nontruncated,
            "eligible_offset": args.eligible_offset,
            "selected_ids": [example.example_id for example in selected_examples],
            "selected_examples_sha256": sha256(args.examples_output),
            "selected_annotations_sha256": sha256(args.annotations_output),
        },
    )


if __name__ == "__main__":
    main()
