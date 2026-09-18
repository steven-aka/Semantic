from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.sequential_policy_evaluation import (
    evaluate_loaded_policy,
    head_state_dict,
)
from src.model.sequential_packet_policy import SequentialPacketPolicy
from src.model.sequential_policy_loss import sequential_policy_loss
from src.reproducibility import experiment_metadata, sha256, write_metadata
from src.training.sequential_policy_data import SequentialHistoryCollator, SequentialHistoryDataset


def checkpoint_key(summary: dict[str, Any]) -> tuple:
    success90 = int(summary["per_level"]["0.9"]["contract_successes"])
    trajectory = float(summary["oracle_cutoff_all_active_contracts_success_fraction"])
    regret = float(summary["mean_oracle_cutoff_ranking_regret_normalized_feasible"])
    gates = (success90 >= 282, trajectory >= 0.90, regret <= 0.03)
    return (
        int(all(gates)),
        sum(gates),
        success90,
        trajectory,
        -regret,
        -float(summary["validation_action_loss"]),
    )


def save_checkpoint(
    model: SequentialPacketPolicy,
    output: Path,
    *,
    step: int,
    summary: dict[str, Any],
    args: argparse.Namespace,
    started: float,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model.lm.save_pretrained(output / "adapter")
    torch.save(head_state_dict(model), output / "sequential_head.pt")
    write_metadata(
        output / "training_metadata.json",
        experiment_metadata(
            stage="v8_one_run_history_aware_sequential_policy",
            base_model=args.model,
            train_data=args.train_data,
            train_sha256=sha256(args.train_data),
            validation_data=args.validation_data,
            validation_sha256=sha256(args.validation_data),
            seed=args.seed,
            optimizer_step=step,
            max_optimizer_steps=args.max_optimizer_steps,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            effective_batch_size=args.batch_size * args.gradient_accumulation,
            lora_learning_rate=args.lora_learning_rate,
            head_learning_rate=args.head_learning_rate,
            progress_weight=args.progress_weight,
            model_dim=args.model_dim,
            beam_width=args.beam_width,
            gradient_checkpointing=args.gradient_checkpointing,
            validation_summary=summary,
            elapsed_seconds=time.perf_counter() - started,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the one-run V8 sequential packet policy")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--model", default="models/Qwen3-4B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-optimizer-steps", type=int, default=750)
    parser.add_argument("--validation-interval-steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--lora-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--progress-weight", type=float, default=0.2)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    if args.max_optimizer_steps <= 0 or args.max_optimizer_steps % args.validation_interval_steps:
        raise ValueError("optimizer budget must be positive and divisible by validation interval")
    if args.batch_size * args.gradient_accumulation != 8:
        raise ValueError("V8 preserves the frozen effective batch size of eight")
    if not torch.cuda.is_available():
        raise RuntimeError("V8 training requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = list(read_jsonl(args.train_data))
    validation_rows = list(read_jsonl(args.validation_data))
    if {row["example_id"] for row in train_rows} & {row["example_id"] for row in validation_rows}:
        raise ValueError("train and validation IDs overlap")

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        get_cosine_schedule_with_warmup,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    lm = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
    )
    lm = prepare_model_for_kbit_training(
        lm, use_gradient_checkpointing=args.gradient_checkpointing
    )
    lm = get_peft_model(
        lm,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )
    lm.config.use_cache = False
    device = torch.device("cuda:0")
    model = SequentialPacketPolicy(lm, model_dim=args.model_dim)
    model.move_head(device=device, dtype=torch.bfloat16)

    train_dataset = SequentialHistoryDataset(train_rows, tokenizer, args.max_sequence_length)
    validation_dataset = SequentialHistoryDataset(
        validation_rows, tokenizer, args.max_sequence_length
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=SequentialHistoryCollator(args.seed),
    )
    lora_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("lm.") and parameter.requires_grad
    ]
    head_parameters = list(model.head_parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": args.lora_learning_rate},
            {"params": head_parameters, "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_optimizer_steps,
    )
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(train_loader)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    history_records = []
    checkpoint_records = []
    window_total = window_action = window_progress = window_accuracy = 0.0
    window_examples = 0
    for optimizer_step in range(1, args.max_optimizer_steps + 1):
        model.train()
        for _ in range(args.gradient_accumulation):
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
                loss = result.total / args.gradient_accumulation
            loss.backward()
            examples = len(batch["example_ids"])
            window_total += float(result.total.detach().item()) * examples
            window_action += float(result.action.detach().item()) * examples
            window_progress += float(result.progress.detach().item()) * examples
            window_accuracy += result.action_accuracy * examples
            window_examples += examples
        torch.nn.utils.clip_grad_norm_([*lora_parameters, *head_parameters], 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if optimizer_step % args.validation_interval_steps == 0:
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=args.validation_batch_size,
                shuffle=False,
                collate_fn=SequentialHistoryCollator(args.seed),
            )
            summary, details = evaluate_loaded_policy(
                model,
                validation_loader,
                exact_dir=args.exact_dir,
                pad_token_id=int(tokenizer.pad_token_id),
                device=device,
                beam_width=args.beam_width,
                progress_weight=args.progress_weight,
            )
            record = {
                "optimizer_step": optimizer_step,
                "train_example_mean_total_loss": window_total / window_examples,
                "train_example_mean_action_loss": window_action / window_examples,
                "train_example_mean_progress_loss": window_progress / window_examples,
                "train_set_action_accuracy": window_accuracy / window_examples,
                **{f"validation_{key}": value for key, value in summary.items() if key != "per_level"},
                "validation_per_level": summary["per_level"],
            }
            history_records.append(record)
            print(json.dumps(record), flush=True)
            checkpoint = output / "checkpoints" / f"step{optimizer_step}"
            save_checkpoint(
                model,
                checkpoint,
                step=optimizer_step,
                summary=summary,
                args=args,
                started=started,
            )
            write_jsonl(checkpoint / "development_details.jsonl", details)
            write_metadata(checkpoint / "development_summary.json", summary)
            checkpoint_records.append(
                {
                    "step": optimizer_step,
                    "checkpoint": str(checkpoint),
                    "summary": summary,
                    "selection_key": list(checkpoint_key(summary)),
                }
            )
            write_jsonl(output / "history.jsonl", history_records)
            window_total = window_action = window_progress = window_accuracy = 0.0
            window_examples = 0

    selected = max(checkpoint_records, key=lambda row: tuple(row["selection_key"]))
    selection = {
        "complete": True,
        "criterion": "all gates, then number passed, fidelity-0.90 successes, trajectory, negative regret, negative action loss",
        "checkpoints": checkpoint_records,
        "selected": selected,
    }
    write_metadata(output / "selected_checkpoint.json", selection)
    summary = selected["summary"]
    success90 = int(summary["per_level"]["0.9"]["contract_successes"])
    trajectory = float(summary["oracle_cutoff_all_active_contracts_success_fraction"])
    regret = float(summary["mean_oracle_cutoff_ranking_regret_normalized_feasible"])
    passed = success90 >= 282 and trajectory >= 0.90 and regret <= 0.03
    decision = {
        "complete": True,
        "decision": "GO_FREEZE_NEW_CONFIRMATION" if passed else "STOP_V8_ONE_RUN",
        "passed": passed,
        "fidelity_0_90_successes": success90,
        "trajectory_success_fraction": trajectory,
        "mean_feasible_regret": regret,
        "requirements": {"fidelity_0_90_successes": 282, "trajectory": 0.90, "regret": 0.03},
        "selected_checkpoint": selected["checkpoint"],
        "locked_roles_used": False,
    }
    write_metadata(output / "development_decision.json", decision)
    write_metadata(
        output / "training_metadata.json",
        experiment_metadata(
            stage="v8_one_run_complete",
            train_data=args.train_data,
            train_sha256=sha256(args.train_data),
            validation_data=args.validation_data,
            validation_sha256=sha256(args.validation_data),
            seed=args.seed,
            elapsed_seconds=time.perf_counter() - started,
            selected_checkpoint=selected["checkpoint"],
            decision=decision["decision"],
        ),
    )
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
