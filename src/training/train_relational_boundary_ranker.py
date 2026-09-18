from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.schemas import read_jsonl, write_jsonl
from src.model.relational_boundary_loss import worst_relational_boundary_loss
from src.model.relational_packet_ranker import RelationalPacketRanker
from src.reproducibility import experiment_metadata, sha256, write_metadata
from src.training.train_atomic_ranker import RankOracleDataset, make_rank_collator


def evaluate(model: RelationalPacketRanker, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    correct = pairs = complete = supervised = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["packet_spans"]
            )["precedence_logits"]
            values = logits.float().cpu()
            for index, preferences in enumerate(batch["preferences"]):
                if not preferences:
                    continue
                losses.append(float(worst_relational_boundary_loss(logits[index : index + 1], [preferences]).item()))
                passed = True
                for winner, loser in preferences:
                    relation = values[index, winner, loser] > 0
                    correct += int(relation)
                    pairs += 1
                    passed &= bool(relation)
                complete += int(passed)
                supervised += 1
    if not losses or not pairs:
        raise ValueError("validation set contains no relational boundaries")
    return {
        "example_mean_worst_boundary_loss": sum(losses) / len(losses),
        "micro_boundary_edge_accuracy": correct / pairs,
        "complete_boundary_edge_separation_fraction": complete / supervised,
        "boundary_pairs": pairs,
        "supervised_examples": supervised,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V4 antisymmetric relational precedence ranker")
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
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("relational ranker training requires CUDA")
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
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    lm = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        quantization_config=quantization, device_map={"": 0},
    )
    lm = prepare_model_for_kbit_training(lm, use_gradient_checkpointing=True)
    lm = get_peft_model(lm, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    lm.config.use_cache = False
    device = torch.device("cuda:0")
    model = RelationalPacketRanker(lm, args.head_hidden_size)
    model.precedence_head.to(device=device, dtype=torch.bfloat16)
    supervised_train = [row for row in train_rows if row["pairwise_preferences"]]
    if not supervised_train:
        raise ValueError("training set has no boundaries")
    collator = make_rank_collator(int(tokenizer.pad_token_id))
    train_loader = DataLoader(
        RankOracleDataset(supervised_train, tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), collate_fn=collator,
    )
    validation_loader = DataLoader(
        RankOracleDataset(validation_rows, tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=False, collate_fn=collator,
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
        for step, batch in enumerate(train_loader, 1):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["packet_spans"]
                )["precedence_logits"]
                boundary_loss = worst_relational_boundary_loss(logits, batch["preferences"])
                loss = boundary_loss / args.gradient_accumulation
            loss.backward()
            loss_sum += float(boundary_loss.detach().item()) * len(batch["example_ids"])
            examples += len(batch["example_ids"])
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation = evaluate(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "train_example_mean_worst_boundary_loss": loss_sum / examples,
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation["example_mean_worst_boundary_loss"] < best_loss:
            best_loss = validation["example_mean_worst_boundary_loss"]
            best_epoch = epoch
            best_state = {
                name: parameter.detach().float().cpu().clone()
                for name, parameter in model.named_parameters() if parameter.requires_grad
            }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, parameter in model.named_parameters():
        if name in best_state:
            parameter.data.copy_(best_state[name].to(parameter.device, parameter.dtype))
    model.lm.save_pretrained(output / "adapter")
    torch.save({name: value.detach().float().cpu() for name, value in model.precedence_head.state_dict().items()}, output / "precedence_head.pt")
    write_jsonl(output / "history.jsonl", history)
    write_metadata(output / "training_metadata.json", experiment_metadata(
        stage="v4_relational_precedence_ranking",
        train_data=args.train_data, train_sha256=sha256(args.train_data),
        validation_data=args.validation_data, validation_sha256=sha256(args.validation_data),
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation, learning_rate=args.learning_rate,
        max_length=args.max_length, head_hidden_size=args.head_hidden_size,
        best_epoch=best_epoch, best_validation_loss=best_loss,
        elapsed_seconds=time.perf_counter() - started,
        objective="example-mean softplus(worst negative stable-boundary antisymmetric edge logit)",
        decoder="exact maximum-weight total order by subset DP",
    ))


if __name__ == "__main__":
    main()
