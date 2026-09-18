from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.sequential_policy_evaluation import load_model
from src.model.sequential_packet_policy import build_sequential_encoder_input
from src.reproducibility import experiment_metadata, sha256, write_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out a frozen V8 policy without Target calls")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="models/Qwen3-4B")
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=8)
    args = parser.parse_args()
    forbidden = ("calibration300", "final_test300", "test300")
    if any(token in args.data.lower() for token in forbidden):
        raise ValueError("locked calibration/final-test roles must not be rolled out")
    rows = list(read_jsonl(args.data))
    tokenizer, model = load_model(args)
    model.eval()
    device = torch.device("cuda:0")
    outputs = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            inputs = [
                build_sequential_encoder_input(
                    tokenizer,
                    row["question"],
                    row["packet_texts"],
                    max_sequence_length=args.max_sequence_length,
                )
                for row in batch
            ]
            packets, questions = model.encode(
                inputs, pad_token_id=int(tokenizer.pad_token_id), device=device
            )
            for index, row in enumerate(batch):
                fractions = torch.tensor(row["packet_tokens"], device=device, dtype=torch.float32)
                fractions /= fractions.sum()
                order = model.beam_order(
                    packets[index], questions[index], fractions, beam_width=args.beam_width
                )
                outputs.append({"example_id": row["example_id"], "decoded_order": list(order)})
            if len(outputs) % 100 < len(batch) or len(outputs) == len(rows):
                print(json.dumps({"processed": len(outputs), "total": len(rows)}), flush=True)
                write_jsonl(args.output, outputs)
    write_metadata(
        f"{args.output}.metadata.json",
        experiment_metadata(
            stage="v8_policy_rollout",
            data=args.data,
            data_sha256=sha256(args.data),
            checkpoint=args.checkpoint,
            checkpoint_metadata_sha256=sha256(Path(args.checkpoint) / "training_metadata.json"),
            beam_width=args.beam_width,
            examples=len(outputs),
            locked_roles_used=False,
        ),
    )


if __name__ == "__main__":
    main()
