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
from src.model.cost_sensitive_policy_loss import cost_sensitive_policy_loss
from src.reproducibility import experiment_metadata, sha256, write_metadata
from src.training.sequential_policy_data import SequentialHistoryCollator, SequentialHistoryDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-backbone V9-A exact-cost probe")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="Selected V8 checkpoint")
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
        raise RuntimeError("V9-A probe requires CUDA")
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
    if not head_parameters or any(not parameter.requires_grad for parameter in head_parameters):
        raise RuntimeError("the complete V8 sequential head must remain trainable")

    train_dataset = SequentialHistoryDataset(train_rows, tokenizer, args.max_sequence_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=SequentialHistoryCollator(args.seed),
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
    window = {
        "total": 0.0,
        "action": 0.0,
        "progress": 0.0,
        "accuracy": 0.0,
        "selected_advantage": 0.0,
        "expected_advantage": 0.0,
        "examples": 0,
    }
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
            result = cost_sensitive_policy_loss(
                action_logits,
                progress_logits,
                batch["action_advantages"],
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
        window["accuracy"] += result.zero_advantage_accuracy * examples
        window["selected_advantage"] += result.mean_selected_advantage * examples
        window["expected_advantage"] += result.mean_expected_advantage * examples
        window["examples"] += examples
        if step % args.log_interval == 0 or step == args.max_optimizer_steps:
            count = window["examples"]
            record = {
                "optimizer_step": step,
                "cost_total_loss": window["total"] / count,
                "cost_action_loss": window["action"] / count,
                "progress_loss": window["progress"] / count,
                "zero_advantage_action_accuracy": window["accuracy"] / count,
                "mean_selected_advantage": window["selected_advantage"] / count,
                "mean_expected_advantage": window["expected_advantage"] / count,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            write_jsonl(output / "training_history.jsonl", history)
            for key in window:
                window[key] = 0.0 if key != "examples" else 0

    torch.save(head_state_dict(model), output / "sequential_head.pt")
    summaries = {}
    if not args.skip_evaluation:
        validation_dataset = SequentialHistoryDataset(
            validation_rows, tokenizer, args.max_sequence_length
        )
        for beam_width in (1, 8):
            loader = DataLoader(
                validation_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=SequentialHistoryCollator(args.seed),
            )
            summary, details = evaluate_loaded_policy(
                model,
                loader,
                exact_dir=args.exact_dir,
                pad_token_id=int(tokenizer.pad_token_id),
                device=device,
                beam_width=beam_width,
                progress_weight=args.progress_weight,
            )
            summaries[f"beam{beam_width}"] = summary
            write_jsonl(output / f"development_beam{beam_width}_details.jsonl", details)
            write_metadata(output / f"development_beam{beam_width}_summary.json", summary)
            print(json.dumps({"beam_width": beam_width, "summary": summary}), flush=True)
    metadata = experiment_metadata(
        stage="v9a_frozen_backbone_exact_cost_probe",
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
        backbone_and_lora_frozen=True,
        complete_sequential_head_trainable=True,
        decoder_comparison="diagnostic beam widths 1 and 8; not formal model selection",
        locked_roles_used=False,
        elapsed_seconds=time.perf_counter() - started,
        summaries=summaries,
    )
    write_metadata(output / "probe_metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
