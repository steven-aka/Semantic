"""One frozen four-fold relative-to-V8 packet intervention probe."""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.schemas import read_jsonl
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import segments


CFG = Path("configs/v17packet_r2g0_relative_bag_probe.json")
ROOT = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
R1 = Path("results/v2_rank_then_cut/v17packet_r1_label_scale512")
OUT = Path("results/v2_rank_then_cut/v17packet_r2g0_relative_bag_probe")


def fold(q: str) -> int:
    return int.from_bytes(hashlib.sha256(q.encode()).digest()[:8], "big") % 4


def label(base: dict, action: dict) -> int:
    assert all(base["hits"][i] == action["hits"][i] for i in (0, 1, 2, 4))
    if base["hits"][3] and not action["hits"][3]:
        return 0  # break
    if not base["hits"][3] and action["hits"][3]:
        return 2  # repair
    return 1  # unchanged quality, including token-only improvement


def main() -> None:
    cfg = json.loads(CFG.read_text())
    assert cfg["status"] == "FROZEN_TRAIN_SIDE_OOF"
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rows = list(read_jsonl(R1 / "per_query.jsonl"))
    assert len(rows) == 512
    ids = {r["example_id"] for r in rows}
    source = {r["example_id"]: r for r in read_jsonl(ROOT / "data/v10_v12_train_train_clean.jsonl")
              if r["example_id"] in ids}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(
        ROOT / "sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    assert len(source) == len(orders) == 512
    tok = AutoTokenizer.from_pretrained(cfg["encoder_repo"], use_fast=True)
    budgets = cfg["field_token_budgets"]
    token_ids, masks, labels, keys, folds = [], [], [], [], []
    truncations = Counter()
    for row in rows:
        q = row["example_id"]
        packets = source[q]["packet_texts"]
        order = orders[q]
        titles = "; ".join(packets[p].split("\n", 1)[0] for p in order[:8])
        displaced = packets[order[8]]
        parts = segments(packets[order[9]])
        assert len(parts) == len(row["sentence_options"])
        for option, fragment in zip(row["sentence_options"], parts):
            if option["context_tokens"] > row["baseline"]["context_tokens"]:
                continue
            fields = {"query": source[q]["question"], "candidate": fragment,
                      "stay_displaced": displaced, "state_titles": titles}
            pieces = []
            for name in ("query", "candidate", "stay_displaced", "state_titles"):
                raw = tok.encode(fields[name], add_special_tokens=False)
                truncations[name] += len(raw) > budgets[name]
                pieces.append(raw[:budgets[name]])
            built = [tok.cls_token_id]
            for piece in pieces:
                built.extend(piece)
                built.append(tok.sep_token_id)
            assert len(built) <= cfg["max_tokens"]
            attention = [1] * len(built)
            pad = cfg["max_tokens"] - len(built)
            token_ids.append(built + [tok.pad_token_id] * pad)
            masks.append(attention + [0] * pad)
            labels.append(label(row["baseline"], option))
            keys.append((q, option["index"]))
            folds.append(fold(q))
    assert len(keys) == 1441
    inputs = torch.tensor(token_ids, dtype=torch.long)
    attention = torch.tensor(masks, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    fold_tensor = torch.tensor(folds)
    probs = np.zeros((len(keys), 3), dtype=np.float32)
    histories = []
    device = torch.device("cuda:0")
    inference_seconds = 0.0
    for held in range(cfg["folds"]):
        train = torch.nonzero(fold_tensor != held, as_tuple=True)[0]
        valid = torch.nonzero(fold_tensor == held, as_tuple=True)[0]
        counts = torch.bincount(y[train], minlength=3).float()
        assert torch.all(counts > 0)
        weights = (counts.sum() / counts).sqrt()
        weights /= weights.mean()
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["encoder_repo"], num_labels=3).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                weight_decay=cfg["weight_decay"])
        scaler = torch.amp.GradScaler("cuda")
        loader = DataLoader(TensorDataset(inputs[train], attention[train], y[train]),
                            batch_size=cfg["batch_size"], shuffle=True,
                            generator=torch.Generator().manual_seed(seed + held))
        history = {"fold": held, "train_actions": len(train),
                   "valid_actions": len(valid), "class_counts": counts.int().tolist(),
                   "class_weights": weights.tolist(), "epoch_loss": []}
        for epoch in range(cfg["epochs"]):
            model.train(); losses = []
            for input_batch, mask_batch, label_batch in loader:
                input_batch = input_batch.to(device)
                mask_batch = mask_batch.to(device)
                label_batch = label_batch.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(input_ids=input_batch, attention_mask=mask_batch).logits
                    loss = torch.nn.functional.cross_entropy(logits.float(), label_batch,
                                                             weight=weights.to(device))
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite R2G0 loss")
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                losses.append(float(loss.detach()))
            history["epoch_loss"].append(float(np.mean(losses)))
            print(json.dumps({"fold": held, "epoch": epoch+1,
                              "loss": history["epoch_loss"][-1]}), flush=True)
        model.eval()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(valid), cfg["batch_size"]):
                idx = valid[start:start+cfg["batch_size"]]
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(input_ids=inputs[idx].to(device),
                                   attention_mask=attention[idx].to(device)).logits
                probs[idx.numpy()] = logits.float().softmax(-1).cpu().numpy()
        torch.cuda.synchronize()
        history["inference_seconds"] = time.perf_counter() - started
        inference_seconds += history["inference_seconds"]
        histories.append(history)
        del model, opt, scaler
        torch.cuda.empty_cache()

    pmap = {key: p for key, p in zip(keys, probs)}
    outrows = []
    for row in rows:
        q = row["example_id"]
        eligible = [a for a in row["sentence_options"]
                    if a["context_tokens"] <= row["baseline"]["context_tokens"]]
        best = max(eligible, key=lambda a: (float(pmap[q,a["index"]][2] -
                                                  pmap[q,a["index"]][0]), -a["index"])) if eligible else None
        best_prob = pmap[q,best["index"]] if best else None
        outrows.append({"example_id": q, "fold": fold(q), "baseline": row["baseline"],
                        "best_action": best,
                        "best_score": float(best_prob[2]-best_prob[0]) if best else None,
                        "best_class_probabilities": best_prob.tolist() if best else None,
                        "repair_opportunity": any(label(row["baseline"], a) == 2 for a in eligible),
                        "eligible_actions": len(eligible)})
    assert sum(r["repair_opportunity"] for r in outrows) == 39
    ranked = sorted((r for r in outrows if r["best_action"]),
                    key=lambda r: (-r["best_score"], r["example_id"]))
    reports = []
    for budget in cfg["budgets_of_all_512_queries"]:
        selected = {r["example_id"] for r in ranked[:math.ceil(512*budget)]}
        chosen = [r["best_action"] if r["example_id"] in selected else r["baseline"]
                  for r in outrows]
        found = [r for r in outrows if r["example_id"] in selected]
        fold_stats = []
        for f in range(4):
            population = [r for r in outrows if r["fold"] == f and r["best_action"]]
            picked = [r for r in found if r["fold"] == f]
            positives = sum(r["repair_opportunity"] for r in population)
            actual = sum(r["repair_opportunity"] for r in picked)
            fold_stats.append({"fold": f, "eligible": len(population), "opportunities": positives,
                               "selected": len(picked), "found": actual,
                               "random_expected_found": len(picked)*positives/len(population)})
        reports.append({"budget": budget, "switches": len(selected),
                        "repair_opportunities_found": sum(r["repair_opportunity"] for r in found),
                        "random_expected_found": len(found)*39/len(ranked),
                        "repairs": sum(not r["baseline"]["hits"][3] and a["hits"][3]
                                       for r,a in zip(outrows,chosen)),
                        "breaks": sum(r["baseline"]["hits"][3] and not a["hits"][3]
                                      for r,a in zip(outrows,chosen)),
                        "success_090": sum(a["hits"][3] is True for a in chosen),
                        "complete": sum(a["complete"] for a in chosen),
                        "five_anchor_success": [sum(a["hits"][i] is True for a in chosen)
                                                for i in range(5)],
                        "mean_cumulative_context_tokens": sum(a["context_tokens"] for a in chosen)/512,
                        "folds": fold_stats})
    def passes(report):
        return (sum(f["found"] > f["random_expected_found"] for f in report["folds"]) >= 3
                and report["repairs"] > report["breaks"]
                and report["complete"] >= 273
                and report["mean_cumulative_context_tokens"] <= 2227.728515625)
    passed = any(passes(r) for r in reports if r["budget"] in (0.05, 0.1))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "oof.jsonl").open("w") as handle:
        for row in outrows:
            handle.write(json.dumps(row) + "\n")
    summary = {"protocol": cfg["protocol"], "queries": 512, "eligible_actions": len(keys),
               "label_counts": dict(Counter(labels)), "field_truncation_counts": dict(truncations),
               "fold_history": histories, "inference_gpu_seconds_total": inference_seconds,
               "inference_gpu_ms_per_query": inference_seconds*1000/512,
               "baseline_090": 381, "baseline_complete": 273,
               "baseline_mean_cumulative_context_tokens": 2227.728515625,
               "policies": reports,
               "primary_gate": "GO_R2G1_LABEL_SCALE" if passed else "STOP_R2G0_FROZEN_PROBE",
               "new_target_calls": 0, "checkpoint_saved": False, "sealed_sets_read": False,
               "limitations": "One train-side design-exposed OOF probe; GPU timing includes shared-device contention and is not a full deployment cost benchmark."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"primary_gate": summary["primary_gate"], "policies": reports,
                      "inference_gpu_ms_per_query": summary["inference_gpu_ms_per_query"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
