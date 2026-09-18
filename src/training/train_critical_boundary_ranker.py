from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.schemas import read_jsonl, write_jsonl
from src.model.atomic_packet_ranker import AtomicPacketRanker
from src.model.critical_boundary_loss import worst_boundary_rank_loss
from src.reproducibility import experiment_metadata, sha256, write_metadata
from src.training.train_atomic_ranker import RankOracleDataset, make_rank_collator


def evaluate_boundary_ranker(
    model: AtomicPacketRanker, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    example_losses = []
    correct = pairs = successful_examples = supervised_examples = 0
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["packet_spans"],
            )
            scores = output["scores"]
            scores_cpu = scores.float().cpu()
            for batch_index, preferences in enumerate(batch["preferences"]):
                if not preferences:
                    continue
                loss = worst_boundary_rank_loss(
                    scores[batch_index : batch_index + 1], [preferences]
                )
                example_losses.append(float(loss.item()))
                example_correct = True
                for winner, loser in preferences:
                    value = scores_cpu[batch_index, winner] > scores_cpu[batch_index, loser]
                    correct += int(value)
                    pairs += 1
                    example_correct &= bool(value)
                successful_examples += int(example_correct)
                supervised_examples += 1
    if not example_losses or not pairs:
        raise ValueError("validation set contains no critical-boundary pairs")
    return {
        "example_mean_worst_boundary_loss": sum(example_losses) / len(example_losses),
        "micro_boundary_pair_accuracy": correct / pairs,
        "complete_boundary_separation_fraction": successful_examples
        / supervised_examples,
        "boundary_pairs": pairs,
        "supervised_examples": supervised_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train scalar packet ranker on the worst stable critical boundary"
    )
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--model", default="models/Qwen3-1.7B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--head-hidden-size", type=int, default=256)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("critical-boundary training requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows = list(read_jsonl(args.train_data))
    validation_rows = list(read_jsonl(args.validation_data))
    if {row["example_id"] for row in train_rows} & {
        row["example_id"] for row in validation_rows
    }:
        raise ValueError("boundary train and validation sets overlap")

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = None
    if not args.no_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    lm = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map={"": 0},
    )
    if quantization is not None:
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
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    lm.config.use_cache = False
    device = torch.device("cuda:0")
    model = AtomicPacketRanker(lm, head_hidden_size=args.head_hidden_size)
    model.rank_head.to(device=device, dtype=torch.bfloat16)

    supervised_train_rows = [row for row in train_rows if row["pairwise_preferences"]]
    if not supervised_train_rows:
        raise ValueError("training set contains no critical-boundary pairs")
    train_data = RankOracleDataset(supervised_train_rows, tokenizer, args.max_length)
    validation_data = RankOracleDataset(validation_rows, tokenizer, args.max_length)
    collator = make_rank_collator(int(tokenizer.pad_token_id))
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_data,
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
        loss_sum = examples = 0
        for step, batch in enumerate(train_loader, start=1):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["packet_spans"],
                )
                boundary_loss = worst_boundary_rank_loss(
                    output["scores"], batch["preferences"]
                )
                loss = boundary_loss / args.gradient_accumulation
            loss.backward()
            loss_sum += float(boundary_loss.detach().item()) * len(batch["example_ids"])
            examples += len(batch["example_ids"])
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation = evaluate_boundary_ranker(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "train_example_mean_worst_boundary_loss": loss_sum / examples,
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if validation["example_mean_worst_boundary_loss"] < best_loss:
            best_loss = validation["example_mean_worst_boundary_loss"]
            best_epoch = epoch
            best_state = {
                name: parameter.detach().float().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, parameter in model.named_parameters():
        if name in best_state:
            parameter.data.copy_(best_state[name].to(parameter.device, parameter.dtype))
    model.lm.save_pretrained(output_dir / "adapter")
    torch.save(
        {
            name: value.detach().float().cpu()
            for name, value in model.rank_head.state_dict().items()
        },
        output_dir / "rank_head.pt",
    )
    write_jsonl(output_dir / "history.jsonl", history)
    write_metadata(
        output_dir / "training_metadata.json",
        experiment_metadata(
            stage="v3_set_valued_critical_boundary_ranking",
            train_data=args.train_data,
            train_data_sha256=sha256(args.train_data),
            validation_data=args.validation_data,
            validation_data_sha256=sha256(args.validation_data),
            base_model=args.model,
            train_examples=len(train_rows),
            train_supervised_examples=len(supervised_train_rows),
            train_zero_boundary_examples=len(train_rows) - len(supervised_train_rows),
            validation_examples=len(validation_rows),
            loss="example-balanced softplus of the worst stable critical-boundary score violation; zero added margin",
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            seed=args.seed,
            lora_rank=32,
            head_hidden_size=args.head_hidden_size,
            checkpoint_selection="minimum consumed-development validation example-mean worst-boundary loss",
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
            elapsed_seconds=time.perf_counter() - started,
        ),
    )


if __name__ == "__main__":
    main()
