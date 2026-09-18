"""CPU-only design audit; never loads a model or updates parameters."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from src.data.schemas import ExactSearchResult, read_jsonl
from src.evaluation.v10_mask_value_decision import decide_v10_probe
from src.search.atomic_nested_chain import best_binary_nested_chain
from src.search.predicted_mask_trajectory import best_order_from_predicted_attainment
from src.search.rank_then_cut import best_prefix_nested_chain


def oracle_control(payload):
    source, exact_dir = payload
    values = source["mask_values"]
    deployment_levels = source["active_levels"]
    levels = source["attainable_levels"]
    order = best_order_from_predicted_attainment(
        [v["attained_levels"] for v in values],
        [v["tokens"] for v in values], len(deployment_levels),
    )
    exact = list(read_jsonl(Path(exact_dir) / (source["example_id"] + ".jsonl"), ExactSearchResult))
    oracle = best_binary_nested_chain(exact, levels)
    learned = best_prefix_nested_chain(exact, order, levels)
    cost = sum(r["tokens"] for r in learned if r["feasible"])
    optimum = sum(r["tokens"] for r in oracle)
    decreases = sum(
        values[mask | (1 << bit)]["tokens"] < values[mask]["tokens"]
        for mask in range(4096) for bit in range(12) if not mask & (1 << bit)
    )
    return {
        "example_id": source["example_id"],
        "all_anchors": all(r["feasible"] for r in learned),
        "cost_matches": cost == optimum,
        "cost_difference": cost - optimum,
        "decreasing_token_edges": decreases,
    }


def role_statistics(path):
    counts = Counter()
    positives = [0] * 5
    denominators = [0] * 5
    sample_pos = [0] * 5
    sample_den = [0] * 5
    weights = []
    for row in read_jsonl(path):
        n = len(row["attainable_levels"])
        counts[n] += 1
        weights.append(5 * len(row["mask_values"]))
        for level in range(5):
            denominators[level] += sum(row["class_population"].values())
            positives[level] += sum(v for k, v in row["class_population"].items() if int(k) > level)
            sample_den[level] += len(row["mask_values"])
            sample_pos[level] += sum(v["attained_levels"] > level for v in row["mask_values"])
    return {
        "examples_by_true_active_count": dict(counts),
        "fixed_anchor_positive_fraction_population": [a / b for a, b in zip(positives, denominators)],
        "fixed_anchor_positive_fraction_sample": [a / b for a, b in zip(sample_pos, sample_den)],
        "loss_terms_per_example_min": min(weights),
        "loss_terms_per_example_max": max(weights),
    }


def counterexamples():
    # Same predictions and token costs, different oracle-supplied active count.
    predicted = [0, 4, 5, 0]
    tokens = [0, 1, 10, 11]
    true_attained = [0, 4, 0, 0]
    def reach(order):
        mask = best = 0
        for bit in order:
            mask |= 1 << bit
            best = max(best, true_attained[mask])
        return best
    four = best_order_from_predicted_attainment([min(a, 4) for a in predicted], tokens, 4)
    five = best_order_from_predicted_attainment(predicted, tokens, 5)
    malformed = {
        "complete": False,
        "examples": 400,
        "per_level": {"0.9": {"examples": 400, "contract_successes": 282}},
        "oracle_cutoff_all_active_contracts_success_fraction": 0.91,
        "mean_oracle_cutoff_ranking_regret_normalized_feasible": 0.02,
    }
    gate = decide_v10_probe(malformed)
    missing_regret = dict(malformed, mean_oracle_cutoff_ranking_regret_normalized_feasible=None)
    null_result = decide_v10_probe(missing_regret)
    return {
        "active_count_inference_leak": {
            "predicted_attainment": predicted, "tokens": tokens,
            "true_attainment": true_attained,
            "order_with_true_active_count_4": four,
            "order_with_fixed_count_5": five,
            "true_reached_with_oracle_count": reach(four),
            "true_reached_with_fixed_count": reach(five),
        },
        "gate_incomplete_282_of_400": gate,
        "gate_no_feasible_trajectory_regret_null": null_result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v10_mask_value_probe.json")
    parser.add_argument("--output", default="results/v2_rank_then_cut/v10_pretraining_audit.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    development = list(read_jsonl(config["data"]["development"]))
    result = {
        "training_started": False,
        "role": "consumed development and train only; ground-truth planner control, not learned-model evaluation",
        "train_statistics": role_statistics(config["data"]["train"]),
        "development_statistics": role_statistics(config["data"]["development"]),
        "counterexamples": counterexamples(),
    }
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(oracle_control, ((r, config["data"]["exact_dir"]) for r in development), chunksize=2))
    result["ground_truth_value_positive_control"] = {
        "examples": len(rows),
        "all_anchor_successes": sum(r["all_anchors"] for r in rows),
        "exact_oracle_cost_matches": sum(r["cost_matches"] for r in rows),
        "decreasing_token_edges": sum(r["decreasing_token_edges"] for r in rows),
        "mismatches": [r for r in rows if not r["all_anchors"] or not r["cost_matches"]],
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
