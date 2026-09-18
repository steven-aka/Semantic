from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.mask_value_evaluation import CachedMaskValueEncoder, evaluate_mask_value_policy
from src.evaluation.v10_mask_value_decision import decide_v10_probe
from src.model.mask_value_head import MaskValueHead, ordinal_mask_value_loss
from src.reproducibility import experiment_metadata, sha256, write_metadata
from src.training.mask_value_data import CachedMaskValueCollator, CachedMaskValueDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-encoder V10 mask-value probe")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--protocol-config", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="models/Qwen3-4B")
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--max-optimizer-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--mask-layers", type=int, default=2)
    parser.add_argument("--mask-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--evaluation-mask-chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol_config).read_text())
    if protocol.get("status") != "APPROVED_TO_RUN":
        raise ValueError(
            "refusing to train unless protocol status is APPROVED_TO_RUN; "
            f"found {protocol.get('status')!r}"
        )
    expected = {
        "train_data": protocol["data"]["train"],
        "validation_data": protocol["data"]["development"],
        "exact_dir": protocol["data"]["exact_dir"],
        "checkpoint": protocol["initialization"]["checkpoint"],
        "train_cache": protocol["data"]["train_cache"],
        "validation_cache": protocol["data"]["development_cache"],
        "output_dir": protocol["output"],
        "model": protocol["initialization"]["model"],
        "model_dim": protocol["model"]["model_dim"],
        "max_sequence_length": protocol["model"]["max_sequence_length"],
        "max_optimizer_steps": protocol["optimization"]["optimizer_steps"],
        "batch_size": protocol["optimization"]["batch_size"],
        "learning_rate": protocol["optimization"]["learning_rate"],
        "weight_decay": protocol["optimization"]["weight_decay"],
        "warmup_steps": protocol["optimization"]["warmup_steps"],
        "mask_layers": protocol["model"]["mask_transformer_layers"],
        "mask_heads": protocol["model"]["attention_heads"],
        "dropout": protocol["model"]["dropout"],
        "evaluation_mask_chunk_size": protocol["evaluation"]["mask_chunk_size"],
        "seed": protocol["optimization"]["seed"],
    }
    mismatches = {
        key: {"argument": getattr(args, key), "protocol": value}
        for key, value in expected.items() if getattr(args, key) != value
    }
    if mismatches:
        raise ValueError(f"training arguments differ from frozen protocol: {mismatches}")
    if not torch.cuda.is_available():
        raise RuntimeError("V10 probe requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_metadata(output / "protocol_snapshot.json", protocol)
    started = time.perf_counter()

    train_rows = list(read_jsonl(args.train_data))
    validation_rows = list(read_jsonl(args.validation_data))
    device = torch.device("cuda:0")
    # Decouple the new head initialization from any RNG use while loading the
    # quantized base model and PEFT adapter.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Keep optimizer parameters and moments in FP32. Autocast below still uses
    # BF16 kernels, while avoiding low-precision AdamW updates on the new head.
    head = MaskValueHead(
        model_dim=args.model_dim,
        layers=args.mask_layers,
        heads=args.mask_heads,
        dropout=args.dropout,
    ).to(device=device)
    train_dataset = CachedMaskValueDataset(train_rows, args.train_cache)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=CachedMaskValueCollator(),
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_optimizer_steps,
    )
    iterator = iter(loader)
    history = []
    window_loss = 0.0
    window_states = 0
    for step in range(1, args.max_optimizer_steps + 1):
        head.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = head(
                batch["packets"].to(device),
                batch["questions"].to(device),
                batch["masks"].to(device),
                batch["example_indices"].to(device),
            )
            loss = ordinal_mask_value_loss(
                logits,
                batch["attained_levels"].to(device),
                batch["sampling_probabilities"].to(device),
                batch["example_indices"].to(device),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        states = len(batch["masks"])
        window_loss += float(loss.detach().item()) * states
        window_states += states
        if step % args.log_interval == 0 or step == args.max_optimizer_steps:
            record = {
                "optimizer_step": step,
                "ordinal_mask_value_loss": window_loss / window_states,
                "sampled_states": window_states,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            write_jsonl(output / "training_history.jsonl", history)
            window_loss = 0.0
            window_states = 0

    torch.save({key: value.detach().float().cpu() for key, value in head.state_dict().items()}, output / "mask_value_head.pt")
    summary = None
    decision = None
    if not args.skip_evaluation:
        validation_dataset = CachedMaskValueDataset(validation_rows, args.validation_cache)
        encoder = CachedMaskValueEncoder(validation_dataset.packets, validation_dataset.questions)
        summary, details = evaluate_mask_value_policy(
            encoder,
            head,
            validation_rows,
            list(range(len(validation_rows))),
            exact_dir=args.exact_dir,
            pad_token_id=0,
            device=device,
            encoder_batch_size=args.batch_size,
            mask_chunk_size=args.evaluation_mask_chunk_size,
        )
        write_jsonl(output / "development_details.jsonl", details)
        write_metadata(output / "development_summary.json", summary)
        decision = decide_v10_probe(summary)
        write_metadata(output / "decision.json", decision)
        print(json.dumps(summary), flush=True)
        print(json.dumps(decision), flush=True)
    metadata = experiment_metadata(
        stage="v10_frozen_encoder_mask_value_probe",
        protocol_config=args.protocol_config,
        protocol_config_sha256=sha256(args.protocol_config),
        seed=args.seed,
        base_checkpoint=args.checkpoint,
        base_checkpoint_metadata_sha256=sha256(Path(args.checkpoint) / "training_metadata.json"),
        train_data=args.train_data,
        train_sha256=sha256(args.train_data),
        validation_data=args.validation_data,
        validation_sha256=sha256(args.validation_data),
        train_cache=args.train_cache,
        train_cache_sha256=sha256(args.train_cache),
        validation_cache=args.validation_cache,
        validation_cache_sha256=sha256(args.validation_cache),
        optimizer_steps=args.max_optimizer_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        scheduler="cosine",
        gradient_clip=1.0,
        approximate_epochs=(args.max_optimizer_steps * args.batch_size) / len(train_rows),
        encoder_frozen_and_cached=True,
        mask_layers=args.mask_layers,
        mask_heads=args.mask_heads,
        dropout=args.dropout,
        trainable_parameters=sum(parameter.numel() for parameter in head.parameters()),
        prediction_threshold=0.5,
        decoder="exact DP over predicted ordinal mask-attainment lattice",
        locked_roles_used=False,
        elapsed_seconds=time.perf_counter() - started,
        summary=summary,
        decision=decision,
    )
    write_metadata(output / "probe_metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
