"""Train-only OOF linear residual editor over four fixed-stop outcomes."""
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
from src.evaluation.v17traj_a0_robust_chain_oracle_audit import exact
from src.evaluation.v17traj_a2_stop_aligned_local_oracle import local_orders, path
from src.evaluation.v17traj_a3_0_local_edit_label_audit import fixed_outcome
from src.training.train_v17h0_multianchor_cutoff import LEVELS


def canonical_choices(base):
    rank = {packet: index for index, packet in enumerate(base)}
    groups = {}
    for order in local_orders(base):
        mask = path(order)[10]
        positions = tuple(rank[packet] for packet in order)
        inversions = sum(positions[i] > positions[j] for i in range(12) for j in range(i + 1, 12))
        key = (inversions, positions)
        if mask not in groups or key < groups[mask][0]:
            groups[mask] = (key, order)
    stay_mask = path(base)[10]
    choices = [tuple(base)] + [groups[mask][1] for mask in sorted(groups) if mask != stay_mask]
    if len(choices) != 4:
        raise ValueError("primary schedule must have four outcome classes")
    return choices


def action_feature(base, alternative, packet_embeddings, query_embedding, packet_tokens):
    base_mask = path(base)[10]
    edit_mask = path(alternative)[10]
    removed = [packet for packet in base if (base_mask >> packet & 1) and not (edit_mask >> packet & 1)]
    added = [packet for packet in base if (edit_mask >> packet & 1) and not (base_mask >> packet & 1)]
    if len(removed) != 1 or len(added) != 1:
        raise ValueError("nonlocal stop-mask edit")
    old, new = removed[0], added[0]
    q = F.normalize(query_embedding.float(), dim=0)
    p_old = F.normalize(packet_embeddings[old].float(), dim=0)
    p_new = F.normalize(packet_embeddings[new].float(), dim=0)
    delta = p_new - p_old
    length_scale = max(sum(packet_tokens), 1)
    position = {packet: index for index, packet in enumerate(base)}
    scalar = torch.tensor([
        float(q @ p_old), float(q @ p_new), float(q @ delta),
        packet_tokens[old] / length_scale, packet_tokens[new] / length_scale,
        (packet_tokens[new] - packet_tokens[old]) / length_scale,
        position[old] / 11, position[new] / 11,
    ], dtype=torch.float32)
    return torch.cat((delta, q * delta, scalar))


def training_label(outcome, baseline):
    hits, cost = outcome
    before, base_cost = baseline
    repair = any(hit is True and old is False for hit, old in zip(hits, before))
    broken = any(hit is False and old is True for hit, old in zip(hits, before))
    if broken:
        return -1
    if cost <= base_cost and (repair or cost < base_cost):
        return 1
    if not repair and cost > base_cost:
        return -1
    return 0


def summarize(rows, source):
    chosen = [row[source] for row in rows]
    complete = sum(result["complete"] for result in chosen)
    anchors = [sum(result["hits"][i] is True for result in chosen) for i in range(5)]
    repair = [sum(row[source]["hits"][i] is True and row["v8"]["hits"][i] is False for row in rows) for i in range(5)]
    breaks = [sum(row[source]["hits"][i] is False and row["v8"]["hits"][i] is True for row in rows) for i in range(5)]
    both = [row for row in rows if row[source]["complete"] and row["v8"]["complete"]]
    return {"queries": len(rows), "anchor_success": anchors, "complete": complete,
            "anchor_repairs": repair, "anchor_breaks": breaks,
            "complete_repairs": sum(row[source]["complete"] and not row["v8"]["complete"] for row in rows),
            "complete_breaks": sum(not row[source]["complete"] and row["v8"]["complete"] for row in rows),
            "edited_queries": sum(row["choice"] != 0 for row in rows) if source == "learned" else 0,
            "mean_normalized_context": mean(result["context_fraction"] for result in chosen),
            "mean_paired_cumulative_token_delta": mean(row[source]["cumulative_tokens"] - row["v8"]["cumulative_tokens"] for row in rows),
            "mean_paired_token_delta_both_complete": mean(row[source]["cumulative_tokens"] - row["v8"]["cumulative_tokens"] for row in both) if both else None}


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "candidates", "data", "rollouts", "embeddings", "exact-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_TRAIN_ONLY_CV":
        raise ValueError("protocol not frozen")
    source = [row for row in read_jsonl(args.candidates) if fold(row["example_id"]) != 4]
    if len(source) != 1421:
        raise ValueError("unexpected query population")
    data = {row["example_id"]: row for row in read_jsonl(args.data)}
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    embedding_cache = torch.load(args.embeddings, map_location="cpu", weights_only=True)
    cache_index = {query: index for index, query in enumerate(embedding_cache["example_ids"])}
    torch.set_num_threads(4)
    features, labels, records = [], [], []
    schedules = {"primary": config["primary_schedule"], **config["secondary_schedules"]}
    for number, row in enumerate(source, 1):
        query = row["example_id"]
        base = orders[query]
        choices = canonical_choices(base)
        fidelity, tokens = exact(Path(args.exact_dir) / f"{query}.jsonl")
        active = set(float(level) for level in row["attainable_levels"])
        if active not in (set(LEVELS[:4]), set(LEVELS)) or tokens[4095] != row["full_tokens"]:
            raise ValueError("invalid source contract")
        idx = cache_index[query]
        packet_embeddings = embedding_cache["packets"][idx]
        question_embedding = embedding_cache["questions"][idx]
        packet_tokens = data[query]["packet_tokens"]
        features.append(torch.stack([action_feature(base, edit, packet_embeddings, question_embedding, packet_tokens)
                                     for edit in choices[1:]]))
        masks = [path(order) for order in choices]
        outcomes = {name: [fixed_outcome(mask, fidelity, tokens, depths, active) for mask in masks]
                    for name, depths in schedules.items()}
        labels.append([training_label(outcomes["primary"][i], outcomes["primary"][0]) for i in range(1, 4)])
        normalized = {}
        for name, results in outcomes.items():
            normalized[name] = [{"hits": list(hits), "complete": all(hit is not False for hit in hits),
                                 "cumulative_tokens": cost,
                                 "context_fraction": cost / (len(active) * tokens[4095])} for hits, cost in results]
        records.append({"example_id": query, "fold": stable_fold(query, 4), "outcomes": normalized})
        if number % 100 == 0:
            print(json.dumps({"prepared": number, "total": len(source)}), flush=True)
    x = torch.stack(features).float()
    y = torch.tensor(labels, dtype=torch.float32)
    assignments = torch.tensor([row["fold"] for row in records])
    decisions, fold_info = [None] * len(records), []
    opt = config["optimizer"]
    for held_out in range(4):
        train_indices = torch.where(assignments != held_out)[0]
        valid_indices = torch.where(assignments == held_out)[0]
        train_x = x[train_indices]
        train_y = y[train_indices]
        mu = train_x.reshape(-1, x.shape[-1]).mean(0)
        sigma = train_x.reshape(-1, x.shape[-1]).std(0).clamp_min(1e-5)
        normalized_train = (train_x - mu) / sigma
        normalized_valid = (x[valid_indices] - mu) / sigma
        torch.manual_seed(opt["seed"] + held_out)
        random.seed(opt["seed"] + held_out)
        model = torch.nn.Linear(x.shape[-1], 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])
        for step in range(opt["steps"]):
            sampled = torch.randint(len(train_indices), (opt["batch_queries"],))
            xb, yb = normalized_train[sampled], train_y[sampled]
            chosen = yb != 0
            scores = model(xb).squeeze(-1)
            loss = F.softplus(-yb[chosen] * scores[chosen]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            valid_scores = model(normalized_valid).squeeze(-1)
            combined = torch.cat((torch.zeros((len(valid_indices), 1)), valid_scores), dim=1)
            choices = combined.argmax(dim=1).tolist()
        for index, choice in zip(valid_indices.tolist(), choices):
            decisions[index] = choice
        fold_info.append({"fold": held_out, "train_queries": len(train_indices), "validation_queries": len(valid_indices),
                          "train_positive_outcomes": int((train_y == 1).sum()),
                          "train_negative_outcomes": int((train_y == -1).sum()),
                          "final_training_loss": float(loss.detach())})
        print(json.dumps(fold_info[-1]), flush=True)
    if any(choice is None for choice in decisions):
        raise AssertionError("missing OOF decision")
    detailed = {name: [] for name in schedules}
    for row, choice in zip(records, decisions):
        for name in schedules:
            outcomes = row["outcomes"][name]
            detailed[name].append({"example_id": row["example_id"], "fold": row["fold"], "choice": choice,
                                   "v8": outcomes[0], "learned": outcomes[choice]})
    summary = {"protocol": config["protocol"], "queries": len(records), "folds": fold_info,
               "label_counts": {"positive": int((y == 1).sum()), "negative": int((y == -1).sum()), "ignored": int((y == 0).sum())},
               "schedules": {}, "primary_gate": None,
               "limitations": ["Train-only query-grouped OOF design validation, not independent lineage-clean confirmation.",
                               "The editor observes only frozen V8 features and packet token lengths; exact fidelity is used exclusively for training labels and OOF evaluation."]}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in detailed.items():
        write_jsonl(output / f"{name}_oof_queries.jsonl", rows)
        summary["schedules"][name] = {"v8": summarize(rows, "v8"), "learned": summarize(rows, "learned")}
    v8 = summary["schedules"]["primary"]["v8"]
    learned = summary["schedules"]["primary"]["learned"]
    gain = any(a > b for a, b in zip(learned["anchor_success"], v8["anchor_success"])) or learned["complete"] > v8["complete"] or learned["mean_normalized_context"] < v8["mean_normalized_context"]
    passes = all(a >= b for a, b in zip(learned["anchor_success"], v8["anchor_success"])) and learned["complete"] >= v8["complete"] and learned["mean_normalized_context"] <= v8["mean_normalized_context"] and gain
    summary["primary_gate"] = "GO_TRAIN_ONLY_PARETO_RESEARCH" if passes else "STOP_A3_1_NO_OOF_PARETO"
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
