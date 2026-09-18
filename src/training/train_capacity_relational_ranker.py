from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from src.data.schemas import read_jsonl, write_jsonl
from src.model.relational_packet_ranker import RelationalPacketRanker
from src.model.tail_cost_relational_loss import tail_cost_relational_loss
from src.reproducibility import experiment_metadata, sha256, write_metadata
from src.training.train_atomic_ranker import RankOracleDataset, make_rank_collator


def make_tail_cost_collator(pad_token_id: int):
    base = make_rank_collator(pad_token_id)

    def collate(items: Sequence[tuple[dict[str, Any], Any]]) -> dict[str, Any]:
        batch = base(items)
        rows = [item[0] for item in items]
        for name in ("safety_preferences", "rate_preferences"):
            batch[name] = [
                [
                    (int(edge["winner"]), int(edge["loser"]), float(edge["weight"]))
                    for edge in row[name]
                ]
                for row in rows
            ]
        return batch

    return collate


def evaluate(
    model: RelationalPacketRanker,
    loader: DataLoader,
    device: torch.device,
    *,
    tail_fraction: float,
    rate_lambda: float,
) -> dict[str, float]:
    model.eval()
    total_losses = []
    safety_losses = []
    rate_losses = []
    safety_correct = safety_pairs = safety_complete = safety_examples = 0
    rate_correct = rate_pairs = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["packet_spans"],
            )["precedence_logits"]
            values = logits.float().cpu()
            for index in range(len(batch["example_ids"])):
                result = tail_cost_relational_loss(
                    logits[index : index + 1],
                    [batch["safety_preferences"][index]],
                    [batch["rate_preferences"][index]],
                    tail_fraction=tail_fraction,
                    rate_lambda=rate_lambda,
                )
                total_losses.append(float(result.total.item()))
                if result.safety_tail is not None:
                    safety_losses.append(float(result.safety_tail.item()))
                if result.rate is not None:
                    rate_losses.append(float(result.rate.item()))
                safety = batch["safety_preferences"][index]
                if safety:
                    passed = True
                    for winner, loser, _ in safety:
                        correct = bool(values[index, winner, loser] > 0)
                        safety_correct += int(correct)
                        safety_pairs += 1
                        passed &= correct
                    safety_complete += int(passed)
                    safety_examples += 1
                for winner, loser, _ in batch["rate_preferences"][index]:
                    rate_correct += int(values[index, winner, loser] > 0)
                    rate_pairs += 1
    if not total_losses or not safety_pairs or not rate_pairs:
        raise ValueError("validation set lacks required tail-cost supervision")
    return {
        "example_mean_total_loss": sum(total_losses) / len(total_losses),
        "example_mean_safety_tail_loss": sum(safety_losses) / len(safety_losses),
        "example_mean_rate_loss": sum(rate_losses) / len(rate_losses),
        "micro_safety_edge_accuracy": safety_correct / safety_pairs,
        "complete_safety_separation_fraction": safety_complete / safety_examples,
        "micro_rate_edge_accuracy": rate_correct / rate_pairs,
        "safety_pairs": safety_pairs,
        "rate_pairs": rate_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V6 backbone-capacity relational ranker")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--model", default="models/Qwen3-1.7B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--head-hidden-size", type=int, default=256)
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--rate-lambda", type=float, default=0.25)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("tail-cost ranker training requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = list(read_jsonl(args.train_data))
    validation_rows = list(read_jsonl(args.validation_data))
    if {row["example_id"] for row in train_rows} & {row["example_id"] for row in validation_rows}:
        raise ValueError("train and validation overlap")

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    lm = prepare_model_for_kbit_training(lm, use_gradient_checkpointing=True)
    lm = get_peft_model(
        lm,
        LoraConfig(
            r=32,
            lora_alpha=64,
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
    model = RelationalPacketRanker(lm, args.head_hidden_size)
    model.precedence_head.to(device=device, dtype=torch.bfloat16)
    train_rows = [
        row for row in train_rows
        if row["safety_preferences"] or row["rate_preferences"]
    ]
    validation_rows = [
        row for row in validation_rows
        if row["safety_preferences"] or row["rate_preferences"]
    ]
    collator = make_tail_cost_collator(int(tokenizer.pad_token_id))
    train_loader = DataLoader(
        RankOracleDataset(train_rows, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        RankOracleDataset(validation_rows, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = safety_sum = rate_sum = 0.0
        examples = safety_batches = rate_batches = 0
        for step, batch in enumerate(train_loader, 1):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["packet_spans"],
                )["precedence_logits"]
                result = tail_cost_relational_loss(
                    logits,
                    batch["safety_preferences"],
                    batch["rate_preferences"],
                    tail_fraction=args.tail_fraction,
                    rate_lambda=args.rate_lambda,
                )
                loss = result.total / args.gradient_accumulation
            loss.backward()
            batch_examples = len(batch["example_ids"])
            loss_sum += float(result.total.detach().item()) * batch_examples
            examples += batch_examples
            if result.safety_tail is not None:
                safety_sum += float(result.safety_tail.detach().item())
                safety_batches += 1
            if result.rate is not None:
                rate_sum += float(result.rate.detach().item())
                rate_batches += 1
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation = evaluate(
            model,
            validation_loader,
            device,
            tail_fraction=args.tail_fraction,
            rate_lambda=args.rate_lambda,
        )
        record = {
            "epoch": epoch,
            "train_example_mean_total_loss": loss_sum / examples,
            "train_batch_mean_safety_tail_loss": safety_sum / safety_batches,
            "train_batch_mean_rate_loss": rate_sum / rate_batches,
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation["example_mean_total_loss"] < best_loss:
            best_loss = validation["example_mean_total_loss"]
            best_epoch = epoch
            best_state = {
                name: parameter.detach().float().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, parameter in model.named_parameters():
        if name in best_state:
            parameter.data.copy_(best_state[name].to(parameter.device, parameter.dtype))
    model.lm.save_pretrained(output / "adapter")
    torch.save(
        {name: value.detach().float().cpu() for name, value in model.precedence_head.state_dict().items()},
        output / "precedence_head.pt",
    )
    write_jsonl(output / "history.jsonl", history)
    write_metadata(
        output / "training_metadata.json",
        experiment_metadata(
            stage="v6_backbone_capacity_relational_ranking",
            base_model=args.model,
            train_data=args.train_data,
            train_sha256=sha256(args.train_data),
            validation_data=args.validation_data,
            validation_sha256=sha256(args.validation_data),
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            head_hidden_size=args.head_hidden_size,
            tail_fraction=args.tail_fraction,
            rate_lambda=args.rate_lambda,
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
            elapsed_seconds=time.perf_counter() - started,
            objective="example-balanced top-quartile weighted safety softplus plus 0.25 weighted cost-sensitive rate softplus",
            decoder="unchanged exact maximum-weight total order by subset DP",
        ),
    )


if __name__ == "__main__":
    main()
