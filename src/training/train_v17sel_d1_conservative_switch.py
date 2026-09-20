"""One frozen baseline-relative candidate-policy run on clean-train queries."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean

import torch

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl, write_jsonl
from src.model.topk_set_selector import TopKSetSelector, successful_set_loss
from src.reproducibility import sha256, write_metadata
from src.training.train_v17sel_c1_set_selector import load_features
from src.training.train_v17h0_multianchor_cutoff import LEVELS


def preferred_actions(row: dict) -> list[int]:
    actions = [a for a in row["actions"] if a["unique_order"]]
    complete = [a for a in actions if a["value_class"] == "repair" and
                not row["anchor"]["complete"] and a["complete"]]
    if complete:
        both = [a for a in complete if row["anchor"]["success"][3] is False and a["success"][3] is True]
        return [a["rank"] for a in (both or complete)]
    repair_090 = [a for a in actions if a["value_class"] == "repair" and
                  row["anchor"]["success"][3] is False and a["success"][3] is True]
    if repair_090:
        return [a["rank"] for a in repair_090]
    saving = [a for a in actions if a["value_class"] == "safe_token_saving"]
    if saving:
        minimum = min(a["oracle_legal_cumulative_tokens"] for a in saving)
        return [a["rank"] for a in saving if a["oracle_legal_cumulative_tokens"] == minimum]
    return [0]


def selected_metrics(rows: list[dict], decisions: list[int]) -> dict:
    if len(rows) != len(decisions):
        raise ValueError("decision population mismatch")
    selected = [row["candidates"][rank] for row, rank in zip(rows, decisions)]
    successes = [bool(c["success"][3]) for c in selected]
    complete = [bool(c["complete"]) for c in selected]
    return {
        "queries": len(rows),
        "per_anchor": [
            {"level": level,
             "eligible": sum(level in row["attainable_levels"] for row in rows),
             "success": sum(bool(c["success"][i]) for row, c in zip(rows, selected)
                            if level in row["attainable_levels"])}
            for i, level in enumerate(LEVELS)
        ],
        "success_090": sum(successes), "complete": sum(complete),
        "mean_penalized_090_token_fraction": mean(c["earliest_tokens"][3] / row["full_tokens"] if ok else 1.0
                                                     for row, c, ok in zip(rows, selected, successes)),
        "mean_earliest_090_tokens_on_success": mean(c["earliest_tokens"][3] for c, ok in zip(selected, successes) if ok)
                                                if any(successes) else None,
        "mean_legal_cumulative_tokens_on_complete": mean(c["oracle_ordered_complete_cumulative_tokens"] for c, ok in zip(selected, complete) if ok)
                                                    if any(complete) else None,
        "selected_rank_histogram": dict(sorted(Counter(decisions).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--value-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_TRAIN_ONCE":
        raise ValueError("protocol not frozen")
    rows, features, unique = load_features(args.candidates, args.cache)
    audit = list(read_jsonl(args.value_audit))
    if len(rows) != 2032 or [r["example_id"] for r in rows] != [r["example_id"] for r in audit]:
        raise ValueError("D0B/feature population mismatch")
    if any(r["fold"] != fold(r["example_id"]) for r in audit):
        raise ValueError("D0B fold mismatch")
    train_ids = [i for i, row in enumerate(rows) if fold(row["example_id"]) != 4]
    exposed_ids = [i for i, row in enumerate(rows) if fold(row["example_id"]) == 4]
    if (len(train_ids), len(exposed_ids)) != (1421, 611):
        raise ValueError("unexpected split")
    positive = torch.zeros((len(rows), 11), dtype=torch.bool)
    for index, item in enumerate(audit):
        positive[index, preferred_actions(item)] = True
        if not bool((positive[index] & unique[index]).any()) or bool((positive[index] & ~unique[index]).any()):
            raise ValueError(f"invalid preferred action set: {item['example_id']}")

    opt = config["optimization"]
    random.seed(opt["seed"])
    torch.manual_seed(opt["seed"])
    torch.cuda.manual_seed_all(opt["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TopKSetSelector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt["steps"])
    generator = torch.Generator().manual_seed(opt["seed"])
    order = torch.randperm(len(train_ids), generator=generator).tolist()
    cursor = 0
    history = []
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for step in range(1, opt["steps"] + 1):
        batch = []
        while len(batch) < opt["batch_size"]:
            if cursor == len(order):
                order = torch.randperm(len(train_ids), generator=generator).tolist()
                cursor = 0
            take = min(opt["batch_size"] - len(batch), len(order) - cursor)
            batch.extend(train_ids[position] for position in order[cursor:cursor + take])
            cursor += take
        indices = torch.tensor(batch)
        model.train()
        scores = model(features[indices].to(device), unique[indices].to(device))
        loss = successful_set_loss(scores, positive[indices].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step in (1, 100, 200, opt["steps"]):
            history.append({"step": step, "loss": float(loss.detach())})
            print(json.dumps(history[-1]), flush=True)

    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, out / "step300.pt")
    write_jsonl(out / "history.jsonl", history)
    model.eval()
    decisions = []
    with torch.no_grad():
        for start in range(0, len(exposed_ids), 128):
            ids = exposed_ids[start:start + 128]
            scores = model(features[ids].to(device), unique[ids].to(device))
            decisions.extend(torch.argmax(scores, dim=1).cpu().tolist())
    selected_rows = [rows[i] for i in exposed_ids]
    baseline = [0] * len(decisions)
    rank1 = [1] * len(decisions)
    current = selected_metrics(selected_rows, decisions)
    base = selected_metrics(selected_rows, baseline)
    reference = selected_metrics(selected_rows, rank1)
    repairs = sum(not row["candidates"][0]["success"][3] and row["candidates"][rank]["success"][3]
                  for row, rank in zip(selected_rows, decisions))
    breaks = sum(row["candidates"][0]["success"][3] and not row["candidates"][rank]["success"][3]
                 for row, rank in zip(selected_rows, decisions))
    both_success_delta = [row["candidates"][rank]["earliest_tokens"][3] - row["candidates"][0]["earliest_tokens"][3]
                          for row, rank in zip(selected_rows, decisions)
                          if row["candidates"][0]["success"][3] and row["candidates"][rank]["success"][3]]
    both_complete_delta = [row["candidates"][rank]["oracle_ordered_complete_cumulative_tokens"] -
                           row["candidates"][0]["oracle_ordered_complete_cumulative_tokens"]
                           for row, rank in zip(selected_rows, decisions)
                           if row["candidates"][0]["complete"] and row["candidates"][rank]["complete"]]
    summary = {
        "protocol": config["protocol"], "evaluation_role": "design-exposed C1 fold; not an independent gate",
        "endpoint": opt["endpoint"], "queries": len(decisions),
        "baseline_v8": base, "reference_v13_rank1": reference, "conservative_switch": current,
        "repairs_090_vs_v8": repairs, "breaks_090_vs_v8": breaks,
        "paired_earliest_090_token_delta_when_both_success": mean(both_success_delta) if both_success_delta else None,
        "paired_legal_cumulative_token_delta_when_both_complete": mean(both_complete_delta) if both_complete_delta else None,
        "holdout_581_read": False, "internal300_read": False, "development_read": False,
        "confirmation_read": False, "cutoff_trained": False, "new_target_calls": 0,
        "artifacts": {"config_sha256": sha256(args.config), "candidates_sha256": sha256(args.candidates),
                      "cache_sha256": sha256(args.cache), "value_audit_sha256": sha256(args.value_audit)},
    }
    write_jsonl(out / "exposed_611_choices.jsonl", [
        {"example_id": row["example_id"], "selected_rank": rank}
        for row, rank in zip(selected_rows, decisions)])
    write_metadata(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
