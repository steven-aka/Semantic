"""Cached replay of a frozen closed-loop answer-stability stop rule."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import normalize_list_answer, parse_cached_list_prediction
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.training.train_v17cut_b0_structured_fixed_v8 import prefix_masks
from src.training.train_v17h0_multianchor_cutoff import LEVELS, meets_fidelity


def answer_set(prediction: str) -> frozenset[str]:
    return frozenset(normalize_list_answer(part) for part in parse_cached_list_prediction(prediction))


def evaluate(depths: list[int], prefixes: list[int], exact: dict[int, dict], active: list[bool],
             full_tokens: int) -> dict:
    flags = [meets_fidelity(exact[prefixes[depth]]["fidelity"], level) if active[i] else None
             for i, (depth, level) in enumerate(zip(depths, LEVELS))]
    cost = sum(exact[prefixes[depth]]["tokens"] for i, depth in enumerate(depths) if active[i])
    return {"success": flags, "complete": all(flag is not False for flag in flags),
            "normalized_final_context": cost / (sum(active) * full_tokens),
            "legal_cumulative_context_tokens": cost}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_TRAIN_ONLY_REPLAY":
        raise ValueError("unfrozen answer-stability protocol")
    source = [row for row in read_jsonl(args.candidates) if fold(row["example_id"]) != 4]
    if len(source) != 1421:
        raise ValueError("unexpected train-only count")
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    table = []
    for number, row in enumerate(source, 1):
        query = row["example_id"]
        prefixes = prefix_masks(orders[query])
        needed = {prefixes[7], prefixes[8], prefixes[10]}
        exact = {}
        for record in read_jsonl(Path(args.exact_dir) / f"{query}.jsonl"):
            mask = state_to_mask(record["state"])
            if mask in needed:
                exact[mask] = record
        if set(exact) != needed:
            raise ValueError(f"missing required prefixes: {query}")
        previous = answer_set(exact[prefixes[7]]["prediction"])
        current = answer_set(exact[prefixes[8]]["prediction"])
        choose_early = previous == current and len(current) >= 9
        active = [level in row["attainable_levels"] for level in LEVELS]
        baseline_depths = [10] * 5
        policy_depths = [8, 8, 8, 8, 10] if choose_early else baseline_depths
        baseline = evaluate(baseline_depths, prefixes, exact, active, row["full_tokens"])
        policy = evaluate(policy_depths, prefixes, exact, active, row["full_tokens"])
        calls = [7, 8, 10] if active[4] or not choose_early else [7, 8]
        probe_context_tokens = sum(exact[prefixes[depth]]["tokens"] for depth in calls)
        table.append({"example_id": query, "choose_early": choose_early,
                      "prediction_count_at_8": len(current), "baseline": baseline,
                      "policy": policy, "target_calls": len(calls),
                      "target_context_tokens_lower_bound": probe_context_tokens,
                      "baseline_target_context_tokens_lower_bound": exact[prefixes[10]]["tokens"]})
        if number % 500 == 0:
            print(json.dumps({"loaded": number, "total": len(source)}), flush=True)
    early = [row for row in table if row["choose_early"]]
    summary = {
        "protocol": config["protocol"], "role": "train-only cached Target replay, no new calls",
        "queries": len(table), "early_switch_queries": len(early),
        "baseline_090": sum(bool(row["baseline"]["success"][3]) for row in table),
        "policy_090": sum(bool(row["policy"]["success"][3]) for row in table),
        "baseline_complete": sum(row["baseline"]["complete"] for row in table),
        "policy_complete": sum(row["policy"]["complete"] for row in table),
        "repair_090": sum(not row["baseline"]["success"][3] and row["policy"]["success"][3] for row in table),
        "break_090": sum(row["baseline"]["success"][3] and not row["policy"]["success"][3] for row in table),
        "complete_gain": sum(not row["baseline"]["complete"] and row["policy"]["complete"] for row in table),
        "complete_break": sum(row["baseline"]["complete"] and not row["policy"]["complete"] for row in table),
        "baseline_mean_final_context_fraction": mean(row["baseline"]["normalized_final_context"] for row in table),
        "policy_mean_final_context_fraction": mean(row["policy"]["normalized_final_context"] for row in table),
        "baseline_mean_target_calls": 1.0,
        "policy_mean_target_calls": mean(row["target_calls"] for row in table),
        "baseline_mean_target_context_tokens_lower_bound": mean(row["baseline_target_context_tokens_lower_bound"] for row in table),
        "policy_mean_target_context_tokens_lower_bound": mean(row["target_context_tokens_lower_bound"] for row in table),
        "scope_limit": "cached predictions are single-run and may differ under repeat generation; context-only probe tokens exclude shared prompt and generated output",
        "fold4_611_read": False, "holdout_581_read": False, "internal300_read": False,
        "development_read": False, "confirmation_read": False, "new_target_calls": 0,
        "artifacts": {"config_sha256": sha256(args.config), "candidates_sha256": sha256(args.candidates),
                      "rollouts_sha256": sha256(args.rollouts)},
    }
    write_metadata(args.output, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
