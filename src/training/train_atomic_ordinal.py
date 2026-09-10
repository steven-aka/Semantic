from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.schemas import read_jsonl, write_jsonl
from src.model.atomic_reveal_policy import AtomicRevealPolicy
from src.model.input_builder import AtomicModelInput, build_atomic_model_input, collate_atomic_inputs
from src.reproducibility import experiment_metadata, sha256, write_metadata


class OrdinalTrajectoryDataset(Dataset):
    def __init__(self, rows: Sequence[dict[str, Any]], tokenizer: Any, max_length: int):
        self.rows = list(rows)
        self.inputs = [
            build_atomic_model_input(
                tokenizer, row["question"], row["packet_texts"], max_length=max_length
            )
            for row in self.rows
        ]
        if any(len(row["packet_texts"]) != 12 for row in self.rows):
            raise ValueError("V1 atomic policy requires exactly 12 packets per example")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], AtomicModelInput]:
        return self.rows[index], self.inputs[index]


def make_collator(pad_token_id: int):
    def collate(items: Sequence[tuple[dict[str, Any], AtomicModelInput]]) -> dict[str, Any]:
        rows, model_inputs = zip(*items)
        input_ids, attention_mask, spans = collate_atomic_inputs(model_inputs, pad_token_id)
        levels = torch.tensor(rows[0]["fidelity_levels"], dtype=torch.float32)
        if any(row["fidelity_levels"] != rows[0]["fidelity_levels"] for row in rows):
            raise ValueError("all training rows must use the same fidelity anchors")
        labels = []
        masks = []
        for row in rows:
            row_labels = []
            row_masks = []
            for packet_labels in row["ordinal_labels"]:
                row_labels.append([0.0 if value is None else float(value) for value in packet_labels])
                row_masks.append([value is not None for value in packet_labels])
            labels.append(row_labels)
            masks.append(row_masks)
        return {
            "example_ids": [row["example_id"] for row in rows],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "packet_spans": spans,
            "levels": levels,
            "labels": torch.tensor(labels, dtype=torch.float32),
            "label_mask": torch.tensor(masks, dtype=torch.bool),
        }

    return collate


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def evaluate(
    model: AtomicRevealPolicy,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
) -> dict[str, float]:
    model.eval()
    loss_sum = correct = count = exact = rows = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, device)
            output = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["packet_spans"],
                batch["levels"],
                batch["labels"],
                batch["label_mask"],
                temperature=temperature,
            )
            predicted = output["trajectory_logits"] >= 0
            mask = batch["label_mask"]
            matched = predicted == batch["labels"].bool()
            loss_sum += float(output["loss"].item()) * int(mask.sum().item())
            correct += int((matched & mask).sum().item())
            count += int(mask.sum().item())
            per_row = (matched | ~mask).flatten(1).all(dim=1)
            exact += int(per_row.sum().item())
            rows += len(per_row)
    return {
        "loss": loss_sum / count if count else 0.0,
        "ordinal_accuracy": correct / count if count else 0.0,
        "exact_trajectory_fraction": exact / rows if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train atomic reveal thresholds with ordinal BCE")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="models/Qwen3-1.7B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("V1 training requires CUDA")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation-fraction must be in (0,1)")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = list(read_jsonl(args.data))
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    validation_count = max(1, round(len(rows) * args.validation_fraction))
    validation_ids = set(order[:validation_count])
    train_rows = [row for index, row in enumerate(rows) if index not in validation_ids]
    validation_rows = [row for index, row in enumerate(rows) if index in validation_ids]

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
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )
    lm.config.use_cache = False
    device = torch.device("cuda:0")
    model = AtomicRevealPolicy(lm)
    model.threshold_head.to(device=device, dtype=torch.bfloat16)

    train_data = OrdinalTrajectoryDataset(train_rows, tokenizer, args.max_length)
    validation_data = OrdinalTrajectoryDataset(validation_rows, tokenizer, args.max_length)
    collator = make_collator(int(tokenizer.pad_token_id))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
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
    best_validation_loss = float("inf")
    best_epoch = 0
    best_trainable_state: dict[str, torch.Tensor] = {}
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = supervised = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = _move(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    batch["input_ids"], batch["attention_mask"],
                    batch["packet_spans"], batch["levels"],
                    batch["labels"], batch["label_mask"],
                    temperature=args.temperature,
                )
                loss = output["loss"] / args.gradient_accumulation
            loss.backward()
            tokens = int(batch["label_mask"].sum().item())
            running_loss += float(output["loss"].detach().item()) * tokens
            supervised += tokens
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation = evaluate(model, validation_loader, device, args.temperature)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / supervised,
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if validation["loss"] < best_validation_loss:
            best_validation_loss = validation["loss"]
            best_epoch = epoch
            best_trainable_state = {
                name: parameter.detach().float().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, parameter in model.named_parameters():
        if name in best_trainable_state:
            parameter.data.copy_(
                best_trainable_state[name].to(
                    device=parameter.device, dtype=parameter.dtype
                )
            )
    model.lm.save_pretrained(output / "adapter")
    torch.save(
        {key: value.detach().float().cpu() for key, value in model.threshold_head.state_dict().items()},
        output / "threshold_head.pt",
    )
    write_jsonl(output / "history.jsonl", history)
    write_metadata(
        output / "training_metadata.json",
        experiment_metadata(
            stage="v1_atomic_ordinal_training",
            input_data=args.data,
            input_data_sha256=sha256(args.data),
            base_model=args.model,
            train_examples=len(train_rows),
            validation_examples=len(validation_rows),
            validation_ids=[row["example_id"] for row in validation_rows],
            loss="masked trajectory BCE on sigmoid((c-tau)/temperature)",
            temperature=args.temperature,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            quantization="none" if args.no_4bit else "nf4_double_quant",
            lora_rank=32,
            elapsed_seconds=time.perf_counter() - started,
            checkpoint_selection="minimum validation BCE across fixed epochs",
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
        ),
    )


if __name__ == "__main__":
    main()
