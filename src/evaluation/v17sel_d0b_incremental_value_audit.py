"""Baseline-relative candidate value, using only cached exact Target outcomes."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.build_v17sel_c0_train_candidates import exact_values, outcome
from src.data.schemas import read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata
from src.training.train_v17h0_multianchor_cutoff import LEVELS, meets_fidelity, project


def prefix_masks(order: list[int]) -> list[int]:
    masks = [0]
    for packet in order:
        masks.append(masks[-1] | (1 << packet))
    if len(masks) != 13 or masks[-1] != 4095:
        raise ValueError("invalid 12-packet order")
    return masks


def window_090(masks: list[int], fidelity: list[float]) -> dict:
    flags = [meets_fidelity(fidelity[mask], .9) for mask in masks]
    if not any(flags):
        return {"first_depth": None, "first_run_length": 0,
                "successful_prefix_count": 0, "first_run_min_margin": None}
    first = flags.index(True)
    length = 0
    while first + length < len(flags) and flags[first + length]:
        length += 1
    return {"first_depth": first, "first_run_length": length,
            "successful_prefix_count": sum(flags),
            "first_run_min_margin": min(fidelity[masks[index]] - .9 for index in range(first, first + length))}


def value_class(anchor: dict, candidate: dict, attainable: set[float]) -> str:
    active = [i for i, level in enumerate(LEVELS) if level in attainable]
    lost = any(anchor["success"][i] and not candidate["success"][i] for i in active)
    gained = any(candidate["success"][i] and not anchor["success"][i] for i in active)
    if anchor["complete"] and not candidate["complete"]:
        lost = True
    if not anchor["complete"] and candidate["complete"]:
        gained = True
    if lost:
        return "break" if not gained else "mixed_break"
    if gained:
        return "repair"
    if anchor["complete"] and candidate["complete"]:
        base_cost = anchor["oracle_ordered_complete_cumulative_tokens"]
        next_cost = candidate["oracle_ordered_complete_cumulative_tokens"]
        if base_cost is None or next_cost is None:
            raise ValueError("complete candidate missing legal cumulative cost")
        if next_cost < base_cost:
            return "safe_token_saving"
    return "no_gain"


def summarize(rows: list[dict], ranks: range) -> dict:
    actions = [action for row in rows for action in row["actions"]
               if action["rank"] in ranks and action["unique_order"]]
    by_class = Counter(action["value_class"] for action in actions)
    return {
        "queries": len(rows), "novel_actions": len(actions),
        "novel_action_classes": dict(sorted(by_class.items())),
        "queries_with_repair": sum(any(a["rank"] in ranks and a["unique_order"] and a["value_class"] == "repair"
                                       for a in row["actions"]) for row in rows),
        "queries_with_safe_token_saving": sum(any(a["rank"] in ranks and a["unique_order"] and a["value_class"] == "safe_token_saving"
                                                 for a in row["actions"]) for row in rows),
        "queries_with_any_strong_benefit": sum(any(a["rank"] in ranks and a["unique_order"] and
                                                   a["value_class"] in ("repair", "safe_token_saving")
                                                   for a in row["actions"]) for row in rows),
        "queries_with_only_one_repair_rank": sum(sum(a["rank"] in ranks and a["unique_order"] and
                                                     a["value_class"] == "repair" for a in row["actions"]) == 1
                                                 for row in rows),
        "repair_actions_known_unstable": sum(a["value_class"] == "repair" and a["known_unstable_090_prefix"] for a in actions),
        "mean_repair_first_run_length_090": mean(a["window_090"]["first_run_length"] for a in actions
                                                 if a["value_class"] == "repair" and a["window_090"]["first_depth"] is not None)
                                                 if any(a["value_class"] == "repair" and a["window_090"]["first_depth"] is not None for a in actions) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--g0a-results", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_READ_ONLY":
        raise ValueError("protocol is not frozen")
    rows = list(read_jsonl(args.candidates))
    if len(rows) != 2032 or len({r["example_id"] for r in rows}) != 2032:
        raise ValueError("unexpected candidate population")
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(args.rollouts)}
    if set(orders) != {r["example_id"] for r in rows}:
        raise ValueError("rollout population mismatch")
    unstable = {(r["example_id"], int(r["mask"])) for r in read_jsonl(args.g0a_results)
                if any(item["success"] != r["cached_success"] for item in r["reruns"])}
    audited = []
    for index, row in enumerate(rows, 1):
        query = row["example_id"]
        fidelity, tokens = exact_values(Path(args.exact_dir) / f"{query}.jsonl")
        if tokens[-1] != row["full_tokens"]:
            raise ValueError(f"full token mismatch: {query}")
        base_order = orders[query]
        attainable = {float(level) for level in row["attainable_levels"]}
        calculated = []
        for candidate in row["candidates"]:
            rank = candidate["rank"]
            order = base_order if rank == 0 else project(base_order, candidate["mask"])
            current = outcome(order, fidelity, tokens, attainable)
            if any(current[key] != candidate[key] for key in ("success", "earliest_tokens", "complete", "oracle_ordered_complete_cumulative_tokens")):
                raise ValueError(f"cached candidate mismatch: {query} rank {rank}")
            masks = prefix_masks(order)
            calculated.append((current, masks, window_090(masks, fidelity)))
        anchor = calculated[0][0]
        actions = []
        for candidate, (current, masks, window) in zip(row["candidates"][1:], calculated[1:]):
            rank = candidate["rank"]
            action = {
                "rank": rank, "unique_order": bool(candidate["unique_order"]),
                "value_class": value_class(anchor, current, attainable),
                "success": [current["success"][i] if LEVELS[i] in attainable else None for i in range(len(LEVELS))],
                "complete": current["complete"],
                "oracle_legal_cumulative_tokens": current["oracle_ordered_complete_cumulative_tokens"],
                "cumulative_token_delta_if_both_complete":
                    current["oracle_ordered_complete_cumulative_tokens"] - anchor["oracle_ordered_complete_cumulative_tokens"]
                    if current["complete"] and anchor["complete"] else None,
                "window_090": window,
                "known_unstable_090_prefix": any((query, mask) in unstable for mask in masks),
            }
            actions.append(action)
        audited.append({
            "example_id": query, "fold": fold(query), "full_tokens": row["full_tokens"],
            "attainable_levels": row["attainable_levels"],
            "anchor": {"rank": 0, "success": [anchor["success"][i] if LEVELS[i] in attainable else None for i in range(len(LEVELS))],
                       "complete": anchor["complete"],
                       "oracle_legal_cumulative_tokens": anchor["oracle_ordered_complete_cumulative_tokens"],
                       "window_090": calculated[0][2]},
            "actions": actions,
        })
        if index % 250 == 0:
            print(json.dumps({"audited": index, "total": len(rows)}), flush=True)
    train = [row for row in audited if row["fold"] != 4]
    exposed = [row for row in audited if row["fold"] == 4]
    result = {
        "protocol": config["protocol"], "role": config["evaluation_role"],
        "anchor": config["anchor"],
        "all_clean_train": {"head_rank_1_to_4": summarize(audited, range(1, 5)),
                            "tail_rank_5_to_10": summarize(audited, range(5, 11))},
        "selector_training_fold": {"head_rank_1_to_4": summarize(train, range(1, 5)),
                                   "tail_rank_5_to_10": summarize(train, range(5, 11))},
        "design_exposed_611_fold": {"head_rank_1_to_4": summarize(exposed, range(1, 5)),
                                    "tail_rank_5_to_10": summarize(exposed, range(5, 11))},
        "decision": "DESCRIPTIVE_ONLY_REQUIRE_CAUSAL_SELECTOR_PROTOCOL_BEFORE_TRAINING",
        "holdout_581_read": False, "internal300_read": False, "development_read": False,
        "confirmation_read": False, "new_target_calls": 0,
        "artifacts": {"config_sha256": sha256(args.config), "candidates_sha256": sha256(args.candidates),
                      "rollouts_sha256": sha256(args.rollouts), "g0a_sha256": sha256(args.g0a_results)},
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "per_query.jsonl", audited)
    write_metadata(out / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
