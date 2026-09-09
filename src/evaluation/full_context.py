from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from src.data.schemas import QAExample, read_jsonl, write_jsonl
from src.evaluation.qa_metrics import exact_match, token_f1
from src.representation.token_counter import count_tokens, load_tokenizer
from src.reproducibility import experiment_metadata, write_metadata
from src.target.answer_parser import parse_answer
from src.target.qwen_runner import TargetRunner


def run_full_context(
    examples: Sequence[QAExample], target: Any, tokenizer: Any, batch_size: int = 32
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        questions = [example.question for example in batch]
        contexts = [example.context for example in batch]
        if hasattr(target, "generate_batch"):
            raw_predictions = target.generate_batch(questions, contexts)
            predictions = [parse_answer(value) for value in raw_predictions]
        else:
            predictions = target.answer_batch(questions, contexts)
            raw_predictions = predictions
        for example, raw_prediction, prediction in zip(batch, raw_predictions, predictions):
            f1 = token_f1(prediction, example.answer)
            rows.append(
                {
                    "example_id": example.example_id,
                    "raw_prediction": raw_prediction,
                    "prediction": prediction,
                    "gold": example.answer,
                    "em": exact_match(prediction, example.answer),
                    "f1": f1,
                    "context_tokens": count_tokens(tokenizer, example.context),
                    "eligible_for_supervision": f1 >= 0.8,
                }
            )
    return rows


def cached_full_context_valid(
    examples: Sequence[QAExample],
    rows: Sequence[dict[str, object]],
    eligible_examples: Sequence[QAExample],
    tokenizer: Any,
) -> bool:
    if len(rows) != len(examples):
        return False
    expected_eligible = []
    for example, row in zip(examples, rows):
        try:
            prediction = parse_answer(str(row["raw_prediction"]))
            f1 = token_f1(prediction, example.answer)
            valid = (
                row["example_id"] == example.example_id
                and row["prediction"] == prediction
                and row["gold"] == example.answer
                and abs(float(row["em"]) - exact_match(prediction, example.answer)) <= 1e-12
                and abs(float(row["f1"]) - f1) <= 1e-12
                and int(row["context_tokens"]) == count_tokens(tokenizer, example.context)
                and row["eligible_for_supervision"] is (f1 >= 0.8)
            )
        except (KeyError, TypeError, ValueError):
            return False
        if not valid:
            return False
        if f1 >= 0.8:
            expected_eligible.append(example)
    return list(eligible_examples) == expected_eligible


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen target on full contexts")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--output", default="results/full_context.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--eligible-output",
        help="Optional QA JSONL containing only examples with full-context F1 >= 0.8",
    )
    parser.add_argument("--eligible-limit", type=int)
    parser.add_argument("--experiment-id", default="v0_full_context")
    parser.add_argument("--dataset-revision", default="unknown")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse-valid",
        action="store_true",
        help="validate complete output/eligible caches before loading the target model",
    )
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.model, args.revision)
    examples = list(read_jsonl(args.examples, QAExample))
    if args.limit is not None:
        examples = examples[: args.limit]
    if args.reuse_valid and args.eligible_output:
        output_path = Path(args.output)
        eligible_path = Path(args.eligible_output)
        metadata_path = output_path.with_suffix(".metadata.json")
        if output_path.exists() and eligible_path.exists() and metadata_path.exists():
            try:
                cached_rows = list(read_jsonl(output_path))
                cached_eligible = list(read_jsonl(eligible_path, QAExample))
                if cached_full_context_valid(examples, cached_rows, cached_eligible, tokenizer):
                    print(f"[full-context] validated and reused {len(examples)} examples", flush=True)
                    return
            except (OSError, TypeError, ValueError):
                pass
    target = TargetRunner(
        args.model,
        backend=args.backend,
        revision=args.revision,
        tokenizer=tokenizer,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    rows = run_full_context(examples, target, tokenizer, args.batch_size)
    write_jsonl(args.output, rows)
    write_metadata(
        Path(args.output).with_suffix(".metadata.json"),
        experiment_metadata(
            experiment_id=args.experiment_id,
            seed=args.seed,
            model_name=args.model,
            model_revision=args.revision,
            dataset_revision=args.dataset_revision,
            backend=args.backend,
            batch_size=args.batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            generation_parameters={"do_sample": False, "max_new_tokens": 64, "enable_thinking": False},
            input_path=args.examples,
        ),
    )
    if args.eligible_output:
        eligible = [example for example, row in zip(examples, rows) if row["eligible_for_supervision"]]
        if args.eligible_limit is not None:
            eligible = eligible[: args.eligible_limit]
        write_jsonl(args.eligible_output, eligible)


if __name__ == "__main__":
    main()
