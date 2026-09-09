from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data.schemas import QAExample, read_jsonl, write_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.representation.token_counter import count_tokens, load_tokenizer
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
    rows = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        questions = [example.question for example in batch]
        full_raw = target.generate_batch(questions, [example.context for example in batch])
        empty_raw = target.generate_batch(questions, [""] * len(batch))
        if len(full_raw) != len(batch) or len(empty_raw) != len(batch):
            raise RuntimeError("target returned an unexpected number of QAMPARI outputs")
        for example, full_text, empty_text in zip(batch, full_raw, empty_raw):
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
                    "full_predictions": full_predictions,
                    "full_metrics": full_metrics,
                    "empty_raw_prediction": empty_text,
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
    args = parser.parse_args()
    examples = list(read_jsonl(args.examples, QAExample))
    if args.limit is not None:
        examples = examples[: args.limit]
    annotations = {row["example_id"]: row for row in read_jsonl(args.annotations)}
    if any(example.example_id not in annotations for example in examples):
        raise ValueError("QAMPARI annotation missing for an example")
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
        ),
    )


if __name__ == "__main__":
    main()
