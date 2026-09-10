from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.reproducibility import experiment_metadata, sha256, tree_sha256, write_metadata
from src.search.atomic_nested_chain import best_binary_nested_chain
from src.search.near_optimal_chain_set import (
    identifiable_pair_relations,
    near_optimal_chain_membership,
    summarize_chain_membership,
)
from src.search.rank_then_cut import (
    best_nested_chain_at_counts,
    best_prefix_nested_chain,
)


def learned_order(thresholds: Sequence[float]) -> tuple[int, ...]:
    return tuple(sorted(range(len(thresholds)), key=lambda index: (thresholds[index], index)))


def state_at_level(thresholds: Sequence[float], level: float) -> tuple[int, ...]:
    return tuple(int(level >= threshold) for threshold in thresholds)


def _arm_rows(
    rows: Sequence[dict[str, object]],
    baseline: Sequence[dict[str, object]],
    full_tokens: int,
) -> dict[str, object]:
    anchors = []
    all_success = True
    total_tokens = baseline_tokens = 0
    for row, oracle in zip(rows, baseline):
        success = bool(row["feasible"])
        all_success &= success
        tokens = int(row["tokens"]) if row["tokens"] is not None else None
        oracle_tokens = int(oracle["tokens"])
        if tokens is not None:
            total_tokens += tokens
        baseline_tokens += oracle_tokens
        anchors.append(
            {
                "fidelity_level": float(row["fidelity_level"]),
                "contract_success": success,
                "tokens": tokens,
                "achieved_fidelity": row["achieved_fidelity"],
                "state": row["state"],
                "cutoff_count": row.get("cutoff_count"),
                "oracle_nested_tokens": oracle_tokens,
                "rate_regret_normalized_if_success": (
                    (tokens - oracle_tokens) / full_tokens if success and tokens is not None else None
                ),
            }
        )
    denominator = len(rows) * full_tokens
    return {
        "all_contracts_success": all_success,
        "contract_successes": sum(bool(row["feasible"]) for row in rows),
        "active_anchors": len(rows),
        "trajectory_rate_regret_normalized": (
            (total_tokens - baseline_tokens) / denominator if all_success else None
        ),
        "anchors": anchors,
    }


def _fixed_states_arm(
    exact_by_state: dict[tuple[int, ...], ExactSearchResult],
    states: Sequence[tuple[int, ...]],
    levels: Sequence[float],
) -> list[dict[str, object]]:
    output = []
    for level, state in zip(levels, states):
        row = exact_by_state[state]
        output.append(
            {
                "example_id": row.example_id,
                "fidelity_level": level,
                "feasible": row.fidelity >= level,
                "cutoff_count": sum(state),
                "tokens": row.tokens,
                "state": list(state),
                "achieved_fidelity": row.fidelity,
            }
        )
    return output


def _pair_diagnostics(
    membership: dict[str, object], order: Sequence[int], width: int
) -> dict[str, object]:
    relations = identifiable_pair_relations(membership, width)
    positions = {packet: position for position, packet in enumerate(order)}
    correct = 0
    for (first, second), direction in relations.items():
        prediction = -1 if positions[first] < positions[second] else 1
        correct += int(prediction == direction)
    total_pairs = width * (width - 1) // 2
    bidirectional = inseparable = 0
    masks_by_level = membership["state_masks_by_level"]
    for first in range(width):
        for second in range(first + 1, width):
            if (first, second) in relations:
                continue
            first_before = any(
                bool(mask & (1 << first)) and not bool(mask & (1 << second))
                for masks in masks_by_level for mask in masks
            )
            second_before = any(
                bool(mask & (1 << second)) and not bool(mask & (1 << first))
                for masks in masks_by_level for mask in masks
            )
            if first_before and second_before:
                bidirectional += 1
            else:
                inseparable += 1
    return {
        "identifiable_pairs": len(relations),
        "identifiable_pairwise_correct": correct,
        "identifiable_pairwise_accuracy": correct / len(relations) if relations else None,
        "bidirectionally_ambiguous_pairs": bidirectional,
        "inseparable_or_unrevealed_pairs": inseparable,
        "total_pairs": total_pairs,
    }


def _never_reveal_diagnostics(
    membership: dict[str, object],
    thresholds: Sequence[float],
    exact_by_state: dict[tuple[int, ...], ExactSearchResult],
) -> dict[str, int]:
    terminal_masks = membership["state_masks_by_level"][-1]
    width = len(thresholds)
    max_level = float(membership["active_levels"][-1])
    predicted = {index for index, threshold in enumerate(thresholds) if threshold > max_level}
    robust_never = {
        index
        for index in range(width)
        if all(not mask & (1 << index) for mask in terminal_masks)
    }
    robust_always = {
        index
        for index in range(width)
        if all(mask & (1 << index) for mask in terminal_masks)
    }
    robust_harmful = set()
    for index in robust_never:
        deltas = []
        for mask in terminal_masks:
            base = exact_by_state[tuple((mask >> bit) & 1 for bit in range(width))]
            added_mask = mask | (1 << index)
            added = exact_by_state[tuple((added_mask >> bit) & 1 for bit in range(width))]
            deltas.append(added.fidelity - base.fidelity)
        if deltas and all(delta < -1e-12 for delta in deltas):
            robust_harmful.add(index)
    return {
        "predicted_never": len(predicted),
        "robust_never": len(robust_never),
        "robust_always": len(robust_always),
        "robust_harmful": len(robust_harmful),
        "robust_never_true_positive": len(predicted & robust_never),
        "robust_never_false_positive": len(predicted - robust_never),
        "robust_never_false_negative": len(robust_never - predicted),
        "robust_harmful_true_positive": len(predicted & robust_harmful),
        "robust_harmful_false_negative": len(robust_harmful - predicted),
    }


def _aggregate_arms(rows: Sequence[dict[str, Any]], arm: str) -> dict[str, object]:
    values = [row["arms"][arm] for row in rows]
    anchors = sum(value["active_anchors"] for value in values)
    successes = sum(value["contract_successes"] for value in values)
    regrets = [
        value["trajectory_rate_regret_normalized"]
        for value in values
        if value["trajectory_rate_regret_normalized"] is not None
    ]
    return {
        "active_contract_success_fraction": successes / anchors,
        "all_active_contracts_success_fraction": mean(
            int(value["all_contracts_success"]) for value in values
        ),
        "feasible_trajectory_examples": len(regrets),
        "mean_trajectory_rate_regret_normalized_feasible": mean(regrets) if regrets else None,
    }


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    sources = list(read_jsonl(args.data))
    predictions = {row["example_id"]: row for row in read_jsonl(args.predictions)}
    slacks = tuple(float(value) for value in args.ambiguity_slacks.split(","))
    if not slacks or any(value < 0 for value in slacks):
        raise ValueError("ambiguity slacks must be non-negative")
    outputs = []
    for source in sources:
        example_id = source["example_id"]
        prediction = predictions[example_id]
        thresholds = [float(value) for value in prediction["predicted_thresholds"]]
        levels = [
            float(level)
            for level, active in zip(source["fidelity_levels"], source["anchor_mask"])
            if active
        ]
        exact = list(
            read_jsonl(Path(args.exact_dir) / f"{example_id}.jsonl", ExactSearchResult)
        )
        by_state = {row.state: row for row in exact}
        width = len(thresholds)
        full_tokens = by_state[(1,) * width].tokens
        baseline = best_binary_nested_chain(exact, levels)
        order = learned_order(thresholds)
        raw_states = [state_at_level(thresholds, level) for level in levels]
        counts = [sum(state) for state in raw_states]

        learned_oracle = best_prefix_nested_chain(exact, order, levels)
        oracle_learned = best_nested_chain_at_counts(exact, counts, levels)
        learned_learned = _fixed_states_arm(by_state, raw_states, levels)
        arms = {
            "oracle_order_oracle_cutoff": _arm_rows(baseline, baseline, full_tokens),
            "learned_order_oracle_cutoff": _arm_rows(learned_oracle, baseline, full_tokens),
            "oracle_order_learned_count": _arm_rows(oracle_learned, baseline, full_tokens),
            "learned_order_learned_count": _arm_rows(learned_learned, baseline, full_tokens),
        }

        ambiguity = {}
        for slack in slacks:
            membership = near_optimal_chain_membership(
                exact, levels, normalized_slack=slack
            )
            ambiguity[str(slack)] = {
                **summarize_chain_membership(membership, width),
                "pairwise": _pair_diagnostics(membership, order, width),
                "never_reveal": _never_reveal_diagnostics(
                    membership, thresholds, by_state
                ),
            }
        outputs.append(
            {
                "example_id": example_id,
                "active_levels": levels,
                "predicted_thresholds": thresholds,
                "learned_order": list(order),
                "learned_counts": counts,
                "arms": arms,
                "near_optimal_ambiguity": ambiguity,
            }
        )

    arm_names = list(outputs[0]["arms"])
    summary: dict[str, object] = {
        "complete": True,
        "protocol": "consumed_fresh30_rank_cut_diagnostic_not_new_test",
        "examples": len(outputs),
        "decomposition": {name: _aggregate_arms(outputs, name) for name in arm_names},
        "ambiguity": {},
        "interpretation_rule": {
            "ordering_bad_if": "learned_order_oracle_cutoff violates contracts or has high rate regret",
            "cutoff_bad_if": "oracle_order_learned_count violates contracts or has high rate regret",
            "both_bad_if": "both isolated arms fail",
            "violations_are_not_negative_regret": True,
        },
        "artifacts": {
            "data_sha256": sha256(args.data),
            "predictions_sha256": sha256(args.predictions),
            "exact_tree_sha256": tree_sha256(args.exact_dir),
        },
    }
    for slack in slacks:
        key = str(slack)
        level_values = [row["near_optimal_ambiguity"][key] for row in outputs]
        pair = [value["pairwise"] for value in level_values]
        never = [value["never_reveal"] for value in level_values]
        identifiable = sum(value["identifiable_pairs"] for value in pair)
        correct = sum(value["identifiable_pairwise_correct"] for value in pair)
        tp = sum(value["robust_never_true_positive"] for value in never)
        fp = sum(value["robust_never_false_positive"] for value in never)
        fn = sum(value["robust_never_false_negative"] for value in never)
        htp = sum(value["robust_harmful_true_positive"] for value in never)
        hfn = sum(value["robust_harmful_false_negative"] for value in never)
        summary["ambiguity"][key] = {
            "mean_packet_anchor_ambiguity_fraction": mean(
                value["packet_anchor_ambiguity_fraction"] for value in level_values
            ),
            "mean_robust_excluded_fraction": mean(
                value["robust_excluded_fraction"] for value in level_values
            ),
            "identifiable_pairs": identifiable,
            "identifiable_pairwise_accuracy": correct / identifiable if identifiable else None,
            "bidirectionally_ambiguous_pairs": sum(
                value["bidirectionally_ambiguous_pairs"] for value in pair
            ),
            "inseparable_or_unrevealed_pairs": sum(
                value["inseparable_or_unrevealed_pairs"] for value in pair
            ),
            "robust_never_packets": sum(value["robust_never"] for value in never),
            "robust_never_precision": tp / (tp + fp) if tp + fp else None,
            "robust_never_recall": tp / (tp + fn) if tp + fn else None,
            "robust_harmful_packets": sum(value["robust_harmful"] for value in never),
            "robust_harmful_recall": htp / (htp + hfn) if htp + hfn else None,
        }
    return outputs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose learned packet ordering and cutoff")
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--ambiguity-slacks", default="0,0.005,0.01,0.02")
    args = parser.parse_args()
    rows, summary = evaluate(args)
    write_jsonl(args.output, rows)
    write_metadata(args.summary, summary)
    write_metadata(
        f"{args.output}.metadata.json",
        experiment_metadata(
            stage="consumed_rank_cut_decomposition",
            data=args.data,
            predictions=args.predictions,
            exact_dir=args.exact_dir,
            ambiguity_slacks=args.ambiguity_slacks,
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
