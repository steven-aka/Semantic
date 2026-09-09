from __future__ import annotations

import argparse
from itertools import islice, product
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from src.data.schemas import ExactSearchResult, QAExample, SemanticPacket, read_jsonl, write_jsonl
from src.evaluation.qa_metrics import exact_match, supporting_fact_recall, token_f1
from src.representation.packet_store import PacketStore
from src.representation.state_builder import build_representation
from src.representation.token_counter import count_tokens, load_tokenizer
from src.reproducibility import experiment_metadata, write_metadata
from src.target.qwen_runner import TargetRunner


def enumerate_states(n_units: int) -> Iterator[tuple[int, ...]]:
    if n_units < 0:
        raise ValueError("n_units must be non-negative")
    return product(range(3), repeat=n_units)


def _batches(iterable: Iterable[tuple[int, ...]], size: int) -> Iterator[list[tuple[int, ...]]]:
    iterator = iter(iterable)
    while batch := list(islice(iterator, size)):
        yield batch


def run_exact_search(
    example: QAExample,
    packets: Sequence[SemanticPacket],
    target: Any,
    tokenizer: Any,
    batch_size: int = 729,
) -> list[ExactSearchResult]:
    """Enumerate all 3^N states and send each batch through one target call."""
    if len(example.units) != len(packets):
        raise ValueError("example units and packets must have equal length")
    supporting = [unit.supporting for unit in example.units]
    results: list[ExactSearchResult] = []
    for states in _batches(enumerate_states(len(packets)), batch_size):
        contexts = [build_representation(packets, state) for state in states]
        predictions = target.answer_batch([example.question] * len(states), contexts)
        if len(predictions) != len(states):
            raise RuntimeError("target returned a different number of predictions than states")
        for state, context, prediction in zip(states, contexts, predictions):
            answer_f1 = token_f1(prediction, example.answer)
            fact_recall = supporting_fact_recall(state, supporting)
            results.append(
                ExactSearchResult(
                    example_id=example.example_id,
                    state=state,
                    tokens=count_tokens(tokenizer, context),
                    prediction=prediction,
                    answer_em=exact_match(prediction, example.answer),
                    answer_f1=answer_f1,
                    fact_recall=fact_recall,
                    fidelity=min(answer_f1, fact_recall),
                )
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batched exact-state target inference")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--packet-dir", default="data/packets")
    parser.add_argument("--output-dir", default="results/exact_search")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--batch-size", type=int, default=729)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--experiment-id", default="v0_exact")
    parser.add_argument("--dataset-revision", default="unknown")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.model, args.revision)
    target = TargetRunner(
        args.model,
        backend=args.backend,
        revision=args.revision,
        tokenizer=tokenizer,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    store = PacketStore(args.packet_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(
        output_dir / "metadata.json",
        experiment_metadata(
            experiment_id=args.experiment_id,
            seed=args.seed,
            model_name=args.model,
            model_revision=args.revision,
            dataset_revision=args.dataset_revision,
            backend=args.backend,
            context_length="semantic packets",
            batch_size=args.batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            distributed_executor_backend=("mp" if args.tensor_parallel_size > 1 else "auto"),
            generation_parameters={"do_sample": False, "max_new_tokens": 64, "enable_thinking": False},
            input_path=args.examples,
        ),
    )
    for index, example in enumerate(read_jsonl(args.examples, QAExample)):
        if args.limit is not None and index >= args.limit:
            break
        output_path = output_dir / f"{example.example_id}.jsonl"
        if output_path.exists() and not args.overwrite:
            continue
        results = run_exact_search(example, store.get(example.example_id), target, tokenizer, args.batch_size)
        write_jsonl(output_path, results)



if __name__ == "__main__":
    main()
