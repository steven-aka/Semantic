from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import torch

from src.data.schemas import read_jsonl
from src.evaluation.sequential_policy_evaluation import load_model
from src.reproducibility import experiment_metadata, sha256, tree_sha256, write_metadata
from src.training.mask_value_data import MaskValueDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen V8 embeddings for V10")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default="models/Qwen3-4B")
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("embedding cache requires CUDA")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")
    all_rows = list(read_jsonl(args.data))
    rows = [row for index, row in enumerate(all_rows) if index % args.num_shards == args.shard_index]
    tokenizer, encoder = load_model(args)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    dataset = MaskValueDataset(rows, tokenizer, args.max_sequence_length)
    packets = []
    questions = []
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, len(rows), args.batch_size):
            packet, question = encoder.encode(
                dataset.inputs[start : start + args.batch_size],
                pad_token_id=int(tokenizer.pad_token_id),
                device=device,
            )
            packets.append(packet.detach().to(device="cpu", dtype=torch.bfloat16))
            questions.append(question.detach().to(device="cpu", dtype=torch.bfloat16))
            if start % (20 * args.batch_size) == 0:
                print({"encoded": min(start + args.batch_size, len(rows)), "total": len(rows)}, flush=True)
    payload = {
        "example_ids": [row["example_id"] for row in rows],
        "packets": torch.cat(packets),
        "questions": torch.cat(questions),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    metadata = experiment_metadata(
        stage="v10_frozen_v8_embedding_cache",
        data=args.data,
        data_sha256=sha256(args.data),
        checkpoint=args.checkpoint,
        checkpoint_metadata_sha256=sha256(Path(args.checkpoint) / "training_metadata.json"),
        adapter_tree_sha256=tree_sha256(Path(args.checkpoint) / "adapter"),
        sequential_head_sha256=sha256(Path(args.checkpoint) / "sequential_head.pt"),
        examples=len(rows),
        shape_packets=list(payload["packets"].shape),
        shape_questions=list(payload["questions"].shape),
        dtype=str(payload["packets"].dtype),
        batch_size=args.batch_size,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(device),
        output_sha256=sha256(output),
    )
    write_metadata(args.manifest, metadata)
    print(metadata, flush=True)


if __name__ == "__main__":
    main()
