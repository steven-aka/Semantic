from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.data.schemas import ExactSearchResult, QAExample, read_jsonl, write_jsonl
from src.evaluation.artifact_status import packet_status, validate_exact_rows
from src.evaluation.fact_coverage import (
    _load_selected_raw,
    evaluate_fact_coverage,
    map_facts_to_units,
    raw_supporting_facts,
    validate_fact_coverage,
)
from src.evaluation.rescore_fact_fidelity import rescore_results
from src.representation.packet_store import PacketStore
from src.representation.token_counter import load_tokenizer
from src.reproducibility import experiment_metadata, write_metadata
from src.search.exact_search import run_exact_search
from src.target.qwen_runner import TargetRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fact coverage and exact search with one frozen target load"
    )
    parser.add_argument("--examples", required=True)
    parser.add_argument("--raw", default="data/raw/hotpot_validation.parquet")
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--output-dir", default="results/v0_2")
    parser.add_argument("--model", default="models/Qwen3-8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=729)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-missing-facts", action="store_true")
    parser.add_argument("--experiment-id-prefix", default="v0_2_fact_atomic")
    args = parser.parse_args()

    examples = list(read_jsonl(args.examples, QAExample))
    packet_report = packet_status(args.examples, args.packet_dir)
    if not packet_report["complete"]:
        raise RuntimeError("packet cache is incomplete or stale: " + "; ".join(packet_report["errors"]))
    store = PacketStore(args.packet_dir)
    output_dir = Path(args.output_dir)
    raw_exact_dir = output_dir / "exact_search_raw"
    fact_exact_dir = output_dir / "exact_search"
    raw_exact_dir.mkdir(parents=True, exist_ok=True)
    fact_exact_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = _load_selected_raw(args.raw, {example.example_id for example in examples})

    target: TargetRunner | None = None
    tokenizer: Any | None = None

    def get_target() -> tuple[TargetRunner, Any]:
        nonlocal target, tokenizer
        if target is None:
            tokenizer = load_tokenizer(args.model, args.revision)
            target = TargetRunner(
                args.model,
                revision=args.revision,
                tokenizer=tokenizer,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
            )
        return target, tokenizer

    coverage_path = output_dir / "fact_coverage.jsonl"
    coverage_rows = None
    if coverage_path.exists() and not args.overwrite:
        try:
            candidate = list(read_jsonl(coverage_path))
            if not validate_fact_coverage(
                examples, raw_rows, candidate, allow_missing=args.allow_missing_facts
            ):
                coverage_rows = candidate
        except (OSError, TypeError, ValueError):
            coverage_rows = None
    if coverage_rows is not None:
        print(f"[coverage] validated and reused {len(coverage_rows)} examples", flush=True)
    else:
        packets = {
            example.example_id: store.get(example.example_id) for example in examples
        }
        mapped_fact_count = sum(
            fact["unit_id"] is not None
            for example in examples
            for fact in map_facts_to_units(
                example,
                raw_supporting_facts(raw_rows[example.example_id]),
                allow_missing=args.allow_missing_facts,
            )
        )
        print(f"[coverage] judging {mapped_fact_count * 2} packet/fact pairs", flush=True)
        active_target, _ = get_target()
        coverage_rows = evaluate_fact_coverage(
            examples,
            raw_rows,
            packets,
            active_target,
            allow_missing=args.allow_missing_facts,
        )
        write_jsonl(coverage_path, coverage_rows)
        write_metadata(
            output_dir / "fact_coverage.metadata.json",
            experiment_metadata(
                experiment_id=f"{args.experiment_id_prefix}_coverage",
                model_name=args.model,
                model_revision=args.revision,
                source_examples=args.examples,
                source_packets=args.packet_dir,
                source_raw=args.raw,
                judgment="frozen_target_binary_explicit_support",
                monotonic_full_closure=True,
            ),
        )
        print(f"[coverage] wrote {len(coverage_rows)} examples", flush=True)
    coverage = {row["example_id"]: row for row in coverage_rows}

    metadata = experiment_metadata(
        experiment_id=f"{args.experiment_id_prefix}_exact",
        model_name=args.model,
        model_revision=args.revision,
        source_examples=args.examples,
        source_packets=args.packet_dir,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        fidelity="min(answer_f1, target_judged_gold_fact_recall)",
    )
    if args.overwrite or not (raw_exact_dir / "metadata.json").exists():
        write_metadata(raw_exact_dir / "metadata.json", metadata)
    if args.overwrite or not (fact_exact_dir / "metadata.json").exists():
        write_metadata(
            fact_exact_dir / "metadata.json", {**metadata, "content_fact_rescored": True}
        )
    total = len(examples)
    for index, example in enumerate(examples, start=1):
        raw_path = raw_exact_dir / f"{example.example_id}.jsonl"
        fact_path = fact_exact_dir / f"{example.example_id}.jsonl"
        results = None
        if raw_path.exists() and not args.overwrite:
            try:
                candidate = list(read_jsonl(raw_path, ExactSearchResult))
                if not validate_exact_rows(example, candidate):
                    results = candidate
            except (OSError, TypeError, ValueError):
                results = None
        if results is None:
            print(f"[exact {index}/{total}] run {example.example_id}", flush=True)
            active_target, active_tokenizer = get_target()
            results = run_exact_search(
                example,
                store.get(example.example_id),
                active_target,
                active_tokenizer,
                args.batch_size,
            )
            write_jsonl(raw_path, results)
        expected_fact = rescore_results(results, coverage[example.example_id])
        fact_valid = False
        if fact_path.exists() and not args.overwrite:
            try:
                fact_valid = list(read_jsonl(fact_path, ExactSearchResult)) == expected_fact
            except (OSError, TypeError, ValueError):
                fact_valid = False
        if fact_valid:
            print(f"[exact {index}/{total}] validated and reused {example.example_id}", flush=True)
        else:
            write_jsonl(fact_path, expected_fact)
            print(f"[exact {index}/{total}] wrote {len(expected_fact)} states", flush=True)


if __name__ == "__main__":
    main()
