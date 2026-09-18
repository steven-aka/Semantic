from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.mask_value_evaluation import evaluate_mask_value_policy
from src.evaluation.sequential_policy_evaluation import load_model
from src.evaluation.v10_mask_value_decision import decide_v10_probe
from src.model.mask_value_head import MaskValueHead
from src.reproducibility import experiment_metadata, sha256, write_metadata
from src.training.mask_value_data import MaskValueDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a V10 mask-value head")
    parser.add_argument("--data", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="V8 encoder checkpoint")
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--model", default="models/Qwen3-4B")
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mask-layers", type=int, default=2)
    parser.add_argument("--mask-heads", type=int, default=8)
    parser.add_argument("--mask-chunk-size", type=int, default=256)
    args = parser.parse_args()
    rows = list(read_jsonl(args.data))
    tokenizer, encoder = load_model(args)
    device = torch.device("cuda:0")
    head = MaskValueHead(
        model_dim=args.model_dim, layers=args.mask_layers, heads=args.mask_heads
    )
    state = torch.load(args.head, map_location="cpu", weights_only=True)
    head.load_state_dict(state)
    head.to(device=device)
    dataset = MaskValueDataset(rows, tokenizer, args.max_sequence_length)
    summary, details = evaluate_mask_value_policy(
        encoder,
        head,
        rows,
        dataset.inputs,
        exact_dir=args.exact_dir,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
        encoder_batch_size=args.batch_size,
        mask_chunk_size=args.mask_chunk_size,
    )
    summary["artifacts"] = {
        "data_sha256": sha256(args.data),
        "encoder_checkpoint_metadata_sha256": sha256(Path(args.checkpoint) / "training_metadata.json"),
        "head_sha256": sha256(args.head),
    }
    write_jsonl(args.output, details)
    write_metadata(args.summary, summary)
    write_metadata(f"{args.summary}.decision.json", decide_v10_probe(summary))
    write_metadata(
        f"{args.output}.metadata.json",
        experiment_metadata(stage="v10_mask_value_evaluation", **summary["artifacts"]),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
