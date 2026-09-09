from __future__ import annotations

import argparse
from itertools import islice, product
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from src.data.schemas import ExactSearchResult, QAExample, SemanticPacket, read_jsonl, write_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.representation.packet_store import PacketStore
from src.representation.state_builder import build_representation
from src.representation.token_counter import count_tokens, load_tokenizer
from src.reproducibility import experiment_metadata, write_metadata
from src.target.qampari_runner import QampariTargetRunner


def _batches(iterable: Iterable[tuple[int, ...]], size: int) -> Iterator[list[tuple[int, ...]]]:
    iterator = iter(iterable)
    while batch := list(islice(iterator, size)):
        yield batch


def validate_qampari_exact_rows(
    example: QAExample, rows: Sequence[ExactSearchResult]
) -> list[str]:
    """Validate a complete QAMPARI search artifact before it may be reused."""
    errors: list[str] = []
    expected_states = set(product(range(3), repeat=len(example.units)))
    states = {row.state for row in rows}
    if len(rows) != len(expected_states) or states != expected_states:
        errors.append(
            f"state coverage rows={len(rows)}, unique={len(states)}, "
            f"expected={len(expected_states)}"
        )
    if any(row.example_id != example.example_id for row in rows):
        errors.append("example id mismatch")
    if any(row.tokens < 0 for row in rows):
        errors.append("negative token count")
    for name in ("answer_em", "answer_f1", "fact_recall", "fidelity"):
        if any(not 0.0 <= float(getattr(row, name)) <= 1.0 for row in rows):
            errors.append(f"{name} outside [0,1]")
    if any(abs(row.fidelity - row.answer_f1) > 1e-12 for row in rows):
        errors.append("fidelity/list-F1 mismatch")
    return errors


def run_qampari_exact_search(
    example: QAExample,
    annotation: Mapping[str, Any],
    packets: Sequence[SemanticPacket],
    target: Any,
    tokenizer: Any,
    *,
    batch_size: int = 729,
) -> list[ExactSearchResult]:
    if len(example.units) != 6 or len(packets) != 6:
        raise ValueError("controlled QAMPARI exact search requires exactly six units")
    atoms = annotation["answer_atoms"]
    if len(atoms) != 10:
        raise ValueError("controlled QAMPARI exact search requires exactly ten answer atoms")
    results = []
    for states in _batches(product(range(3), repeat=6), batch_size):
        contexts = [build_representation(packets, state) for state in states]
        raw_predictions = target.generate_batch([example.question] * len(states), contexts)
        if len(raw_predictions) != len(states):
            raise RuntimeError("target returned an unexpected number of list predictions")
        for state, context, raw_prediction in zip(states, contexts, raw_predictions):
            predictions = parse_list_prediction(raw_prediction)
            metrics = qampari_list_metrics(predictions, atoms)
            exact = float(metrics["correct"] == 10 and metrics["predicted"] == 10)
            results.append(
                ExactSearchResult(
                    example_id=example.example_id,
                    state=state,
                    tokens=count_tokens(tokenizer, context),
                    prediction=" # ".join(predictions),
                    answer_em=exact,
                    answer_f1=float(metrics["f1"]),
                    fact_recall=float(metrics["recall"]),
                    fidelity=float(metrics["f1"]),
                )
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled QAMPARI list-F1 exact search")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="models/Qwen3-8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--batch-size", type=int, default=729)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    examples = list(read_jsonl(args.examples, QAExample))
    if args.limit is not None:
        examples = examples[: args.limit]
    annotations = {row["example_id"]: row for row in read_jsonl(args.annotations)}
    tokenizer = load_tokenizer(args.model, args.revision)
    target = QampariTargetRunner(
        args.model,
        revision=args.revision,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    store = PacketStore(args.packet_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(
        output_dir / "metadata.json",
        experiment_metadata(
            stage="m0_qampari_development_exact_search",
            scientific_status="development_only_not_a_locked_result",
            model=args.model,
            model_revision=args.revision,
            fidelity="official_alias_aware_list_f1",
            answer_atoms=10,
            states_per_example=729,
            input_examples=args.examples,
            input_annotations=args.annotations,
            input_packets=args.packet_dir,
        ),
    )
    for index, example in enumerate(examples, start=1):
        output = output_dir / f"{example.example_id}.jsonl"
        if output.exists():
            try:
                cached = list(read_jsonl(output, ExactSearchResult))
                cache_errors = validate_qampari_exact_rows(example, cached)
            except (OSError, TypeError, ValueError) as exc:
                cache_errors = [str(exc)]
            if not cache_errors:
                print(
                    f"[qampari exact {index}/{len(examples)}] reuse {example.example_id}",
                    flush=True,
                )
                continue
            print(
                f"[qampari exact {index}/{len(examples)}] regenerate invalid cache: "
                + "; ".join(cache_errors),
                flush=True,
            )
        rows = run_qampari_exact_search(
            example,
            annotations[example.example_id],
            store.get(example.example_id),
            target,
            tokenizer,
            batch_size=args.batch_size,
        )
        write_jsonl(output, rows)
        print(f"[qampari exact {index}/{len(examples)}] wrote {len(rows)} states", flush=True)


if __name__ == "__main__":
    main()
