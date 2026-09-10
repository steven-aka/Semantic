from __future__ import annotations

import argparse
from dataclasses import replace

from src.data.schemas import QAExample, read_jsonl, write_jsonl
from src.evaluation.qampari_full_context import run_qampari_full_context
from src.representation.atomic_packet_store import AtomicPacketStore
from src.representation.state_builder import build_atomic_representation
from src.representation.token_counter import load_tokenizer
from src.reproducibility import experiment_metadata, write_metadata
from src.target.qampari_runner import QampariTargetRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate empty and all-one binary atomic QAMPARI representations"
    )
    parser.add_argument("--examples", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="models/Qwen3-8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    examples = list(read_jsonl(args.examples, QAExample))
    if args.limit is not None:
        examples = examples[: args.limit]
    annotations = {row["example_id"]: row for row in read_jsonl(args.annotations)}
    store = AtomicPacketStore(args.packet_dir)
    atomic_examples = []
    for example in examples:
        packets = store.get(example.example_id)
        atomic_examples.append(
            replace(
                example,
                context=build_atomic_representation(packets, (1,) * len(packets)),
            )
        )
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
        atomic_examples, annotations, target, tokenizer, batch_size=args.batch_size
    )
    write_jsonl(args.output, rows)
    write_metadata(
        f"{args.output}.metadata.json",
        experiment_metadata(
            stage="m0_qampari_atomic_all_one_ceiling",
            input_examples=args.examples,
            input_annotations=args.annotations,
            input_packets=args.packet_dir,
            model=args.model,
            model_revision=args.revision,
            max_new_tokens=args.max_new_tokens,
            evaluated_context="binary_atomic_all_one_representation",
        ),
    )


if __name__ == "__main__":
    main()
