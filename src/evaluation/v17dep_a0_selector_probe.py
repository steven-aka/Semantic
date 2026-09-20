"""Train-only out-of-fold probe for beneficial baseline deviations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import torch
from torch import nn

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata
from src.training.train_v17sel_c1_set_selector import load_features


def average_precision(pairs: list[tuple[float, bool]]) -> float | None:
    positives = sum(label for _, label in pairs)
    if not positives:
        return None
    ordered = sorted(enumerate(pairs), key=lambda item: (-item[1][0], item[0]))
    hit = 0
    precision = 0.0
    for index, (_, (_, label)) in enumerate(ordered, 1):
        if label:
            hit += 1
            precision += hit / index
    return precision / positives


def train_linear(x: torch.Tensor, y: torch.Tensor, config: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, nn.Linear]:
    mu = x.mean(0)
    sigma = x.std(0).clamp_min(1e-4)
    normalized = ((x - mu) / sigma).to(device)
    labels = y.float().to(device)
    model = nn.Linear(x.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(config["seed"])
    for _ in range(config["epochs"]):
        indices = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), config["batch_size"]):
            batch = indices[start:start + config["batch_size"]].to(device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(normalized[batch]).squeeze(-1), labels[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return mu, sigma, model.eval()


def metrics(selected: list[tuple[dict, int]]) -> dict:
    repair_090 = break_090 = complete_gain = complete_break = benefit = harmful = 0
    paired_complete_delta = []
    width_1 = width_multi = 0
    for row, rank in selected:
        anchor = row["anchor"]
        action = row["actions"][rank - 1]
        benefit += action["value_class"] in ("repair", "safe_token_saving")
        harmful += action["value_class"] in ("break", "mixed_break")
        repair_090 += anchor["success"][3] is False and action["success"][3] is True
        break_090 += anchor["success"][3] is True and action["success"][3] is False
        complete_gain += not anchor["complete"] and action["complete"]
        complete_break += anchor["complete"] and not action["complete"]
        if anchor["complete"] and action["complete"]:
            paired_complete_delta.append(action["cumulative_token_delta_if_both_complete"])
        if action["value_class"] == "repair" and action["success"][3]:
            width_1 += action["window_090"]["first_run_length"] == 1
            width_multi += action["window_090"]["first_run_length"] > 1
    return {"switches": len(selected), "beneficial_actions": benefit, "harmful_actions": harmful,
            "repair_090": repair_090, "break_090": break_090,
            "complete_gain": complete_gain, "complete_break": complete_break,
            "paired_complete_queries": len(paired_complete_delta),
            "mean_paired_legal_cumulative_token_delta": mean(paired_complete_delta) if paired_complete_delta else None,
            "repair_action_first_window_width_1": width_1,
            "repair_action_first_window_width_gt_1": width_multi}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--value-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_TRAIN_ONLY_DIAGNOSTIC":
        raise ValueError("unfrozen protocol")
    rows, features, unique = load_features(args.candidates, args.cache)
    audit = list(read_jsonl(args.value_audit))
    if len(rows) != 2032 or [r["example_id"] for r in rows] != [r["example_id"] for r in audit]:
        raise ValueError("candidate/value population mismatch")
    population = [i for i, row in enumerate(rows) if fold(row["example_id"]) != 4]
    if len(population) != 1421:
        raise ValueError("unexpected train-only population")
    actions = [(i, rank) for i in population for rank in range(5, 11) if bool(unique[i, rank])]
    labels = torch.tensor([audit[i]["actions"][rank - 1]["value_class"] in ("repair", "safe_token_saving")
                           for i, rank in actions], dtype=torch.bool)
    full = torch.stack([features[i, rank] - features[i, 0] for i, rank in actions])
    variants = {"scalar": full[:, -9:], "full": full}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["probe"]["seed"])
    predictions = {}
    summaries = {}
    for name, x in variants.items():
        oof = [None] * len(actions)
        for heldout in range(4):
            train = [position for position, (query_index, _) in enumerate(actions) if fold(rows[query_index]["example_id"]) != heldout]
            test = [position for position, (query_index, _) in enumerate(actions) if fold(rows[query_index]["example_id"]) == heldout]
            if not train or not test:
                raise ValueError("empty selector probe fold")
            mu, sigma, model = train_linear(x[train], labels[train], config["probe"], device)
            with torch.no_grad():
                values = model(((x[test] - mu) / sigma).to(device)).squeeze(-1).cpu().tolist()
            for position, value in zip(test, values):
                oof[position] = float(value)
        if any(value is None for value in oof):
            raise ValueError("incomplete out-of-fold predictions")
        predictions[name] = oof
        summary = {"action_average_precision": average_precision(list(zip(oof, labels.tolist()))),
                   "positive_action_prevalence": float(labels.float().mean()), "actions": len(actions)}
        top_by_query = {}
        for action_index, (query_index, rank) in enumerate(actions):
            item = (oof[action_index], -rank, rank)
            if query_index not in top_by_query or item > top_by_query[query_index]:
                top_by_query[query_index] = item
        budget_results = {}
        for fraction in (.01, .02, .05):
            selected = []
            fixed_rank5 = []
            random_control = []
            for heldout in range(4):
                queries = [i for i in population if fold(rows[i]["example_id"]) == heldout and i in top_by_query]
                budget = max(1, math.floor(len(queries) * fraction))
                top = sorted(queries, key=lambda i: (-top_by_query[i][0], rows[i]["example_id"]))[:budget]
                selected.extend((audit[i], top_by_query[i][2]) for i in top)
                fixed_rank5.extend((audit[i], 5) for i in top if bool(unique[i, 5]))
                random_queries = sorted(queries, key=lambda i: hashlib.sha256(rows[i]["example_id"].encode()).hexdigest())[:budget]
                random_control.extend((audit[i], next(rank for rank in range(5, 11) if bool(unique[i, rank])))
                                      for i in random_queries)
            budget_results[str(fraction)] = {
                "probe_best_tail": metrics(selected),
                "same_trigger_queries_fixed_rank5": metrics(fixed_rank5),
                "hash_query_first_unique_tail": metrics(random_control),
            }
        summary["selective_replay"] = budget_results
        summaries[name] = summary
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "selector_oof_actions.jsonl", [
        {"example_id": rows[i]["example_id"], "rank": rank,
         "beneficial": bool(labels[position]),
         "scalar_logit": predictions["scalar"][position],
         "full_logit": predictions["full"][position]}
        for position, (i, rank) in enumerate(actions)])
    result = {"protocol": config["protocol"], "role": "train-only OOF diagnostic, not deployed selection",
              "queries": len(population), "probes": summaries,
              "fold4_611_read": False, "holdout_581_read": False,
              "internal300_read": False, "development_read": False,
              "confirmation_read": False, "new_target_calls": 0,
              "artifacts": {"config_sha256": sha256(args.config), "candidates_sha256": sha256(args.candidates),
                            "cache_sha256": sha256(args.cache), "value_audit_sha256": sha256(args.value_audit)}}
    write_metadata(out / "selector_summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
