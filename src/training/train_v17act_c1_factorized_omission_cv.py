"""One frozen train-side CV test of explicit packet-pair omission features."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.v17d1b_frozen_boundary_separability import stable_fold
from src.training.train_v17traj_a3_1_boundary_editor_cv import action_feature, canonical_choices, training_label

LEVELS = (0.6, 0.7, 0.8, 0.9, 0.95)


def pair_feature(base, alternative, packets, query, packet_tokens):
    old = action_feature(base, alternative, packets, query, packet_tokens)
    q = F.normalize(query.float(), dim=0)
    def represent(order):
        omitted = [packet for packet in base if packet not in set(order[:10])]
        if len(omitted) != 2:
            raise AssertionError("not a two-packet omission")
        a, b = (F.normalize(packets[packet].float(), dim=0) for packet in omitted)
        total = max(sum(packet_tokens), 1)
        scalar = torch.tensor([float(a @ b), float(q @ (a + b)), float(q @ (a * b)),
                               sum(packet_tokens[packet] for packet in omitted) / total,
                               abs(packet_tokens[omitted[0]] - packet_tokens[omitted[1]]) / total], dtype=torch.float32)
        return torch.cat((a + b, a * b, q * (a + b), q * (a * b), scalar))
    return torch.cat((old, represent(alternative) - represent(base)))


def outcome(record, active):
    hits = tuple(record["f1"] + 1e-6 >= level if level in active else None for level in LEVELS)
    return hits, record["tokens"] * len(active)


def summary(rows, key):
    chosen = [row[key] for row in rows]
    return {"anchor_success": [sum(item["hits"][i] is True for item in chosen) for i in range(5)],
            "complete": sum(all(hit is not False for hit in item["hits"]) for item in chosen),
            "mean_context_fraction": mean(item["context_fraction"] for item in chosen),
            "anchor_repairs": [sum(row[key]["hits"][i] is True and row["stay"]["hits"][i] is False for row in rows) for i in range(5)],
            "anchor_breaks": [sum(row[key]["hits"][i] is False and row["stay"]["hits"][i] is True for row in rows) for i in range(5)],
            "complete_repairs": sum(all(h is not False for h in row[key]["hits"]) and any(h is False for h in row["stay"]["hits"]) for row in rows),
            "complete_breaks": sum(any(h is False for h in row[key]["hits"]) and all(h is not False for h in row["stay"]["hits"]) for row in rows),
            "edited_queries": sum(row["choice"] != 0 for row in rows) if key == "learned" else 0}


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "candidates", "data", "rollouts", "embeddings", "fresh", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg["status"] != "FROZEN_TRAIN_ONLY_CV":
        raise ValueError("unfrozen training protocol")
    source = [row for row in read_jsonl(args.candidates) if fold(row["example_id"]) != 4]
    if len(source) != 1421:
        raise AssertionError("wrong train population")
    data = {row["example_id"]: row for row in read_jsonl(args.data)}
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    fresh = {(row["example_id"], row["mask"]): row for row in read_jsonl(args.fresh)}
    if len(fresh) != 5684:
        raise AssertionError("fresh Target cache incomplete")
    embeddings = torch.load(args.embeddings, map_location="cpu", weights_only=True)
    embedding_index = {query: i for i, query in enumerate(embeddings["example_ids"])}
    xs, labels, records = [], [], []
    torch.set_num_threads(4)
    for number, row in enumerate(source, 1):
        query = row["example_id"]
        base = orders[query]
        actions = canonical_choices(base)
        masks = [sum(1 << packet for packet in action[:10]) for action in actions]
        if len(set(masks)) != 4:
            raise AssertionError("duplicate action context")
        active = {float(level) for level in row["attainable_levels"]}
        values = [fresh[(query, mask)] for mask in masks]
        outcomes = [outcome(value, active) for value in values]
        labels.append([training_label(value, outcomes[0]) for value in outcomes[1:]])
        index = embedding_index[query]
        xs.append(torch.stack([pair_feature(base, action, embeddings["packets"][index], embeddings["questions"][index],
                                            data[query]["packet_tokens"]) for action in actions[1:]]))
        records.append({"example_id": query, "fold": stable_fold(query, 4), "labels": labels[-1],
                        "options": [{"mask": mask, "hits": list(value[0]), "tokens": raw["tokens"],
                                     "context_fraction": raw["tokens"] / row["full_tokens"]}
                                    for mask, value, raw in zip(masks, outcomes, values)]})
        if number % 200 == 0:
            print(json.dumps({"prepared": number, "total": len(source)}), flush=True)
    x = torch.stack(xs).float()
    y = torch.tensor(labels, dtype=torch.float32)
    folds = torch.tensor([row["fold"] for row in records])
    choices = [None] * len(records)
    margins = [None] * len(records)
    fold_results = []
    opt = cfg["optimizer"]
    for heldout in range(4):
        train = torch.where(folds != heldout)[0]
        valid = torch.where(folds == heldout)[0]
        train_x, train_y = x[train], y[train]
        mu = train_x.reshape(-1, x.shape[-1]).mean(0)
        sigma = train_x.reshape(-1, x.shape[-1]).std(0).clamp_min(1e-5)
        train_x = (train_x - mu) / sigma
        valid_x = (x[valid] - mu) / sigma
        torch.manual_seed(opt["seed"] + heldout)
        random.seed(opt["seed"] + heldout)
        model = torch.nn.Linear(x.shape[-1], 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])
        for _ in range(opt["steps"]):
            sampled = torch.randint(len(train), (opt["batch_queries"],))
            xb, yb = train_x[sampled], train_y[sampled]
            used = yb != 0
            scores = model(xb).squeeze(-1)
            loss = F.softplus(-yb[used] * scores[used]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            scores = model(valid_x).squeeze(-1)
            selected = torch.cat((torch.zeros((len(valid), 1)), scores), dim=1).argmax(dim=1).tolist()
        for j, (position, choice) in enumerate(zip(valid.tolist(), selected)):
            choices[position] = choice
            margins[position] = float(scores[j].max())
        fold_results.append({"fold": heldout, "train_queries": len(train), "valid_queries": len(valid),
                             "train_positive": int((train_y == 1).sum()), "train_negative": int((train_y == -1).sum()),
                             "loss": float(loss.detach())})
        print(json.dumps(fold_results[-1]), flush=True)
    if any(value is None for value in choices):
        raise AssertionError("missing OOF choice")
    selected_rows = []
    for row, choice, margin in zip(records, choices, margins):
        selected_rows.append({"example_id": row["example_id"], "fold": row["fold"], "choice": choice,
                              "margin": margin, "chosen_label": row["labels"][choice - 1] if choice else 0,
                              "stay": row["options"][0], "learned": row["options"][choice]})
    baseline = summary(selected_rows, "stay")
    learned = summary(selected_rows, "learned")
    quality = all(a >= b for a, b in zip(learned["anchor_success"], baseline["anchor_success"])) and learned["complete"] >= baseline["complete"]
    context_ok = learned["mean_context_fraction"] <= baseline["mean_context_fraction"]
    strict = any(a > b for a, b in zip(learned["anchor_success"], baseline["anchor_success"])) or learned["complete"] > baseline["complete"] or learned["mean_context_fraction"] < baseline["mean_context_fraction"]
    result = {"protocol": cfg["protocol"], "queries": len(source), "fresh_target_contexts": len(fresh),
              "label_counts": {"positive": int((y == 1).sum()), "negative": int((y == -1).sum()), "ignored": int((y == 0).sum())},
              "folds": fold_results, "stay": baseline, "learned": learned,
              "selected_label_counts": {str(value): sum(row["chosen_label"] == value for row in selected_rows if row["choice"] != 0) for value in (-1, 0, 1)},
              "gate": "GO_ACT_C1_TRAIN_ONLY_PARETO" if quality and context_ok and strict else "STOP_ACT_C1_NO_TRAIN_ONLY_PARETO",
              "limitations": ["Train-only OOF folds use queries seen in historical upstream V8 training.",
                              "Fresh Target labels are one current-contract run, not repeated statistical truth.",
                              "Only fixed-depth10 endpoint quality is tested; no deployment or progressive cutoff claim."]}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "oof_queries.jsonl", selected_rows)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
