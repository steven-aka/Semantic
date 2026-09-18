from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.sequential_policy_evaluation import (
    evaluate_loaded_policy,
    head_state_dict,
    load_model,
)
from src.model.sequential_policy_loss import sequential_policy_loss
from src.reproducibility import experiment_metadata, sha256, write_metadata
from src.training.sequential_policy_data import SequentialHistoryCollator, SequentialHistoryDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-backbone V9-B one-round coverage probe")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="models/Qwen3-4B")
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--max-optimizer-steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--progress-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--max-training-examples", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("V9-B probe requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    train_rows = list(read_jsonl(args.train_data))
    if args.max_training_examples is not None:
        train_rows = train_rows[: args.max_training_examples]
    validation_rows = list(read_jsonl(args.validation_data))
    tokenizer, model = load_model(args)
    device = torch.device("cuda:0")
    for parameter in model.lm.parameters():
        parameter.requires_grad = False
    head_parameters = list(model.head_parameters())
    train_dataset = SequentialHistoryDataset(train_rows, tokenizer, args.max_sequence_length)
    train_collator = SequentialHistoryCollator(
        args.seed, sample_counts={"oracle": 12, "single": 12, "random": 8, "model": 12}
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=train_collator,
    )
    optimizer = torch.optim.AdamW(
        head_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_optimizer_steps,
    )
    iterator = iter(train_loader)
    window = {"total": 0.0, "action": 0.0, "progress": 0.0, "accuracy": 0.0, "examples": 0}
    history = []
    for step in range(1, args.max_optimizer_steps + 1):
        model.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            action_logits, progress_logits, _, _ = model.forward_states(
                batch["inputs"],
                batch["histories"].to(device),
                batch["history_lengths"].to(device),
                batch["example_indices"].to(device),
                batch["packet_token_fractions"].to(device),
                pad_token_id=int(tokenizer.pad_token_id),
                device=device,
            )
            result = sequential_policy_loss(
                action_logits,
                progress_logits,
                batch["optimal_actions"],
                batch["reached_levels"].to(device),
                batch["active_level_counts"].to(device),
                progress_weight=args.progress_weight,
            )
        result.total.backward()
        torch.nn.utils.clip_grad_norm_(head_parameters, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        examples = len(batch["example_ids"])
        window["total"] += float(result.total.detach().item()) * examples
        window["action"] += float(result.action.detach().item()) * examples
        window["progress"] += float(result.progress.detach().item()) * examples
        window["accuracy"] += result.action_accuracy * examples
        window["examples"] += examples
        if step % args.log_interval == 0 or step == args.max_optimizer_steps:
            count = window["examples"]
            record = {
                "optimizer_step": step,
                "total_loss": window["total"] / count,
                "action_loss": window["action"] / count,
                "progress_loss": window["progress"] / count,
                "set_valued_action_accuracy": window["accuracy"] / count,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            write_jsonl(output / "training_history.jsonl", history)
            for key in window:
                window[key] = 0.0 if key != "examples" else 0

    torch.save(head_state_dict(model), output / "sequential_head.pt")
    summary = None
    if not args.skip_evaluation:
        validation_dataset = SequentialHistoryDataset(
            validation_rows, tokenizer, args.max_sequence_length
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=SequentialHistoryCollator(args.seed),
        )
        summary, details = evaluate_loaded_policy(
            model,
            validation_loader,
            exact_dir=args.exact_dir,
            pad_token_id=int(tokenizer.pad_token_id),
            device=device,
            beam_width=8,
            progress_weight=args.progress_weight,
        )
        write_jsonl(output / "development_beam8_details.jsonl", details)
        write_metadata(output / "development_beam8_summary.json", summary)
        print(json.dumps(summary), flush=True)
    metadata = experiment_metadata(
        stage="v9b_frozen_backbone_one_round_coverage_probe",
        seed=args.seed,
        base_checkpoint=args.checkpoint,
        base_checkpoint_metadata_sha256=sha256(Path(args.checkpoint) / "training_metadata.json"),
        train_data=args.train_data,
        train_sha256=sha256(args.train_data),
        validation_data=args.validation_data,
        validation_sha256=sha256(args.validation_data),
        optimizer_steps=args.max_optimizer_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        history_samples={"oracle": 12, "single": 12, "random": 8, "model": 12},
        backbone_and_lora_frozen=True,
        complete_sequential_head_trainable=True,
        locked_roles_used=False,
        elapsed_seconds=time.perf_counter() - started,
        summary=summary,
    )
    write_metadata(output / "probe_metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
