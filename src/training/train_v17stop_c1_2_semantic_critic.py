"""Single frozen train-side 4-fold semantic STOP/CONTINUE experiment."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.schemas import read_jsonl

ROOT = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
C0 = Path("results/v2_rank_then_cut/v17stop_c0_prefix_output_retention")
P0 = Path("results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
OUT = Path("results/v2_rank_then_cut/v17stop_c1_2_semantic_critic")
CONTINUE = "depth10_plus_old_context_supported_depth9"
CLASSES = ("SS", "SF", "FS", "FF")


def fold(q: str) -> int:
    return int.from_bytes(hashlib.sha256(q.encode()).digest()[:8], "big") % 4


def main() -> None:
    cfg = json.loads(Path("configs/v17stop_c1_2_semantic_critic.json").read_text())
    assert cfg["status"] == "FROZEN_TRAIN_SIDE_090_ONLY"
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    data = {r["example_id"]: r for r in read_jsonl(ROOT / "data/v10_v12_train_train_clean.jsonl")}
    c0 = sorted(read_jsonl(C0 / "per_query.jsonl"), key=lambda x: x["example_id"])
    ninth = {}
    for path in sorted(P0.glob("shard_*_of_3/per_prefix.jsonl")):
        for r in read_jsonl(path):
            if r["depth"] == 9:
                ninth[r["example_id"]] = r
    assert len(c0) == 1421 and len({r["example_id"] for r in c0}) == 1421
    texts, labels, folds = [], [], []
    for r in c0:
        q = r["example_id"]
        assert q in data and q in ninth
        d = data[q]
        texts.append(f"[QUERY] {d['question']} [TARGET] 0.90 [ANSWER9] {ninth[q]['prediction']} "
                     f"[NEW_PACKET] {d['packet_texts'][r['added_packet']]}")
        stop = r["outputs"]["depth9_only"]["success"]["0.9"]
        cont = r["outputs"][CONTINUE]["success"]["0.9"]
        labels.append(CLASSES.index(("S" if stop else "F") + ("S" if cont else "F")))
        folds.append(fold(q))
    tok = AutoTokenizer.from_pretrained(cfg["encoder_repo"], use_fast=True)
    encoded = tok(texts, truncation=True, max_length=cfg["max_tokens"],
                  padding="max_length", return_tensors="pt")
    x = encoded["input_ids"]
    mask = encoded["attention_mask"]
    y = torch.tensor(labels, dtype=torch.long)
    probabilities = np.zeros((len(c0), 4), dtype=np.float32)
    histories = []
    device = torch.device("cuda:0")
    for held in range(4):
        train = torch.tensor([i for i, f in enumerate(folds) if f != held])
        valid = torch.tensor([i for i, f in enumerate(folds) if f == held])
        assert len(train) + len(valid) == len(c0) and set(train.tolist()).isdisjoint(valid.tolist())
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["encoder_repo"], num_labels=4).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                weight_decay=cfg["weight_decay"])
        scaler = torch.amp.GradScaler("cuda")
        train_loader = DataLoader(TensorDataset(x[train], mask[train], y[train]),
                                  batch_size=cfg["batch_size"], shuffle=True,
                                  generator=torch.Generator().manual_seed(cfg["seed"] + held))
        history = {"fold": held, "train_queries": len(train), "valid_queries": len(valid), "epoch_loss": []}
        for epoch in range(cfg["epochs"]):
            model.train()
            losses = []
            for ids, attention, target in train_loader:
                ids, attention, target = ids.to(device), attention.to(device), target.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(input_ids=ids, attention_mask=attention, labels=target)
                if not torch.isfinite(out.loss):
                    raise ValueError(f"nonfinite loss fold {held} epoch {epoch}")
                scaler.scale(out.loss).backward()
                scaler.step(opt)
                scaler.update()
                losses.append(float(out.loss.detach()))
            history["epoch_loss"].append(float(np.mean(losses)))
            print(json.dumps({"fold": held, "epoch": epoch + 1,
                              "loss": history["epoch_loss"][-1]}), flush=True)
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(valid), cfg["batch_size"]):
                idx = valid[start:start + cfg["batch_size"]]
                with torch.autocast("cuda", dtype=torch.float16):
                    scores = model(input_ids=x[idx].to(device),
                                   attention_mask=mask[idx].to(device)).logits
                probabilities[idx.numpy()] = scores.float().softmax(-1).cpu().numpy()
        histories.append(history)
        del model, opt, scaler
        torch.cuda.empty_cache()
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-4)
    out_rows = []
    for i, r in enumerate(c0):
        out_rows.append({"example_id": r["example_id"], "fold": folds[i],
                         "label": CLASSES[labels[i]],
                         "probabilities": {name: float(probabilities[i, j]) for j, name in enumerate(CLASSES)}})
    results = []
    for threshold in cfg["thresholds"]:
        chosen = [float(p[2] - p[1]) > threshold for p in probabilities]
        success = [r["outputs"][CONTINUE if go else "depth9_only"]["success"]["0.9"]
                   for r, go in zip(c0, chosen)]
        results.append({"threshold": threshold, "continue_count": sum(chosen),
                        "success_090": sum(success),
                        "mean_target_tokens": float(np.mean([
                            r["depth9_target_tokens"] + (r["depth10_target_tokens"] if go else 0)
                            for r, go in zip(c0, chosen)])),
                        "mean_final_context_tokens": float(np.mean([
                            r["depth10_context_tokens"] if go else r["depth9_context_tokens"]
                            for r, go in zip(c0, chosen)])),
                        "repairs_vs_single_depth10": sum(
                            ok and not r["outputs"]["depth10_only"]["success"]["0.9"]
                            for r, ok in zip(c0, success)),
                        "breaks_vs_single_depth10": sum(
                            not ok and r["outputs"]["depth10_only"]["success"]["0.9"]
                            for r, ok in zip(c0, success))})
    nll = float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-8)).mean())
    report = {"protocol": cfg["protocol"], "population": len(c0),
              "class_counts": {name: labels.count(i) for i, name in enumerate(CLASSES)},
              "fold_history": histories, "oof_nll_diagnostic": nll,
              "fixed_depth10": {"success_090": sum(r["outputs"]["depth10_only"]["success"]["0.9"] for r in c0),
                                "mean_target_tokens": float(np.mean([r["depth10_target_tokens"] for r in c0])),
                                "mean_final_context_tokens": float(np.mean([r["depth10_context_tokens"] for r in c0]))},
              "policies": results, "new_target_calls": 0, "sealed_outcome_sets_read": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUT / "oof.jsonl").open("w") as handle:
        for r in out_rows:
            handle.write(json.dumps(r) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
