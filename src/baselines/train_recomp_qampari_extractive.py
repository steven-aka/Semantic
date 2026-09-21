"""Train a RECOMP-style bi-encoder while keeping the downstream Target absent."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from src.data.schemas import read_jsonl


class Pairs(Dataset):
    def __init__(self, path: Path, seed: int):
        rng = random.Random(seed); self.rows = []
        for row in read_jsonl(path):
            negatives = row["negative_sentences"]
            for i, pos in enumerate(row["positive_sentences"]):
                neg = negatives[(i + rng.randrange(len(negatives))) % len(negatives)]
                self.rows.append((row["question"], pos["text"], neg["text"], row["example_id"]))
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


def pool(hidden, mask):
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)


def encode(model, tokenizer, texts, device, max_length):
    batch = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt").to(device)
    return torch.nn.functional.normalize(pool(model(**batch).last_hidden_state, batch["attention_mask"]), dim=-1)


@torch.no_grad()
def validate(model, tokenizer, loader, device, max_length):
    model.eval(); correct = count = 0; margins = []
    for questions, positives, negatives, _ in loader:
        q = encode(model, tokenizer, questions, device, max_length)
        p = encode(model, tokenizer, positives, device, max_length)
        n = encode(model, tokenizer, negatives, device, max_length)
        margin = (q * p).sum(-1) - (q * n).sum(-1)
        correct += int((margin > 0).sum()); count += len(margin); margins.extend(margin.cpu().tolist())
    return {"pair_accuracy": correct / count, "mean_margin": sum(margins) / len(margins), "pairs": count}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/baseline_recomp_qampari_extractive.json"))
    p.add_argument("--data-dir", type=Path, default=Path("results/v2_rank_then_cut/baseline_recomp_qampari_extractive"))
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/baselines/recomp_qampari_extractive"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args(); cfg = json.loads(args.config.read_text())
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["compressor_initialization"], local_files_only=True)
    model = AutoModel.from_pretrained(cfg["compressor_initialization"], local_files_only=True).to(args.device)
    train = Pairs(args.data_dir / "train.jsonl", cfg["seed"])
    val = Pairs(args.data_dir / "validation.jsonl", cfg["seed"] + 1)
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True, generator=generator)
    val_loader = DataLoader(val, batch_size=cfg["batch_size"] * 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best = None; history = []; args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train(); total = steps = 0
        for questions, positives, negatives, _ in train_loader:
            q = encode(model, tokenizer, questions, args.device, cfg["max_length"])
            pos = encode(model, tokenizer, positives, args.device, cfg["max_length"])
            neg = encode(model, tokenizer, negatives, args.device, cfg["max_length"])
            pos_score = (q * pos).sum(-1); neg_score = (q * neg).sum(-1)
            loss = torch.relu(cfg["margin"] - pos_score + neg_score).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.detach()); steps += 1
        metrics = validate(model, tokenizer, val_loader, args.device, cfg["max_length"])
        metrics.update({"epoch": epoch, "train_loss": total / steps}); history.append(metrics); print(json.dumps(metrics), flush=True)
        key = (metrics["pair_accuracy"], metrics["mean_margin"])
        if best is None or key > best[0]:
            best = (key, metrics); model.save_pretrained(args.output_dir); tokenizer.save_pretrained(args.output_dir)
    result = {"protocol": cfg["protocol"], "train_pairs": len(train), "validation_pairs": len(val),
              "history": history, "selected": best[1], "target_loaded": False, "target_trainable": False,
              "sealed_sets_read": False}
    (args.output_dir / "training_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
