from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data.schemas import QAExample, read_jsonl, write_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.representation.token_counter import count_tokens, load_tokenizer
from src.representation.packet_store import PacketStore
from src.representation.state_builder import build_representation
from src.reproducibility import experiment_metadata, write_metadata
from src.target.qampari_runner import QampariTargetRunner


def run_qampari_full_context(
    examples: Sequence[QAExample],
    annotations: Mapping[str, Mapping[str, Any]],
    target: Any,
    tokenizer: Any,
    *,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    def generate_records(
        questions: Sequence[str], contexts: Sequence[str]
    ) -> list[dict[str, Any]]:
        if hasattr(target, "generate_batch_records"):
            return list(target.generate_batch_records(questions, contexts))
        return [
            {
                "text": text,
                "finish_reason": "unknown",
                "stop_reason": None,
                "generated_tokens": None,
            }
            for text in target.generate_batch(questions, contexts)
        ]

    rows = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        questions = [example.question for example in batch]
        full_records = generate_records(questions, [example.context for example in batch])
        empty_records = generate_records(questions, [""] * len(batch))
        if len(full_records) != len(batch) or len(empty_records) != len(batch):
            raise RuntimeError("target returned an unexpected number of QAMPARI outputs")
        for example, full_record, empty_record in zip(batch, full_records, empty_records):
            full_text = str(full_record["text"])
            empty_text = str(empty_record["text"])
            annotation = annotations[example.example_id]
            atoms = annotation["answer_atoms"]
            full_predictions = parse_list_prediction(full_text)
            empty_predictions = parse_list_prediction(empty_text)
            full_metrics = qampari_list_metrics(full_predictions, atoms)
            empty_metrics = qampari_list_metrics(empty_predictions, atoms)
            rows.append(
                {
                    "example_id": example.example_id,
                    "gold_answers": len(atoms),
                    "context_tokens": count_tokens(tokenizer, example.context),
                    "full_raw_prediction": full_text,
                    "full_finish_reason": full_record["finish_reason"],
                    "full_stop_reason": full_record["stop_reason"],
                    "full_generated_tokens": full_record["generated_tokens"],
                    "full_predictions": full_predictions,
                    "full_metrics": full_metrics,
                    "empty_raw_prediction": empty_text,
                    "empty_finish_reason": empty_record["finish_reason"],
                    "empty_stop_reason": empty_record["stop_reason"],
                    "empty_generated_tokens": empty_record["generated_tokens"],
                    "empty_predictions": empty_predictions,
                    "empty_metrics": empty_metrics,
                    "f1_context_gain": float(full_metrics["f1"]) - float(empty_metrics["f1"]),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="QAMPARI controlled full/empty-context development audit")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="models/Qwen3-8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--packet-dir",
        help="when set, evaluate the actual all-state-2 representation instead of example.context",
    )
    args = parser.parse_args()
    examples = list(read_jsonl(args.examples, QAExample))
    if args.limit is not None:
        examples = examples[: args.limit]
    annotations = {row["example_id"]: row for row in read_jsonl(args.annotations)}
    if any(example.example_id not in annotations for example in examples):
        raise ValueError("QAMPARI annotation missing for an example")
    if args.packet_dir:
        store = PacketStore(args.packet_dir)
        examples = [
            replace(
                example,
                context=build_representation(
                    store.get(example.example_id), (2,) * len(example.units)
                ),
            )
            for example in examples
        ]
    tokenizer = load_tokenizer(args.model, args.revision)
    target = QampariTargetRunner(
        args.model,
        revision=args.revision,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    rows = run_qampari_full_context(
        examples, annotations, target, tokenizer, batch_size=args.batch_size
    )
    write_jsonl(args.output, rows)
    write_metadata(
        Path(args.output).with_suffix(".metadata.json"),
        experiment_metadata(
            stage="m0_qampari_full_context_development_only",
            examples=len(rows),
            model=args.model,
            model_revision=args.revision,
            max_new_tokens=args.max_new_tokens,
            input_examples=args.examples,
            input_annotations=args.annotations,
            input_packets=args.packet_dir,
            evaluated_context="all_state_2_representation" if args.packet_dir else "example_context",
        ),
    )


if __name__ == "__main__":
    main()
