from __future__ import annotations

import argparse
import heapq
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from src.data.schemas import ExactSearchResult, read_jsonl
from src.evaluation.rank_failure_audit import (
    clopper_pearson_lower,
    minimum_successes_for_lower_bound,
)
from src.reproducibility import sha256, write_metadata
from src.search.rank_then_cut import best_prefix_nested_chain, validate_packet_order


def stable_partial_order_projection(
    order: Sequence[int], preferences: Sequence[Sequence[int]]
) -> tuple[int, ...]:
    """Return the closest stable Kahn projection under learned-order tie breaks."""
    order = validate_packet_order(order, len(order))
    position = {packet: index for index, packet in enumerate(order)}
    outgoing = {packet: [] for packet in order}
    indegree = {packet: 0 for packet in order}
    seen = set()
    for pair in preferences:
        if len(pair) != 2:
            raise ValueError("ranking preference must contain winner and loser")
        winner, loser = (int(value) for value in pair)
        if winner not in indegree or loser not in indegree or winner == loser:
            raise ValueError("invalid ranking preference")
        if (winner, loser) in seen:
            continue
        seen.add((winner, loser))
        outgoing[winner].append(loser)
        indegree[loser] += 1
    available = [(position[packet], packet) for packet in order if indegree[packet] == 0]
    heapq.heapify(available)
    projected = []
    while available:
        _, packet = heapq.heappop(available)
        projected.append(packet)
        for loser in outgoing[packet]:
            indegree[loser] -= 1
            if indegree[loser] == 0:
                heapq.heappush(available, (position[loser], loser))
    if len(projected) != len(order):
        raise ValueError("partial-order preferences contain a cycle")
    return tuple(projected)


def never_reveal_tail_projection(
    order: Sequence[int], never_reveal_labels: Sequence[bool | None]
) -> tuple[int, ...]:
    order = validate_packet_order(order, len(order))
    if len(never_reveal_labels) != len(order):
        raise ValueError("never-reveal labels and packet order must align")
    return tuple(
        [packet for packet in order if never_reveal_labels[packet] is not True]
        + [packet for packet in order if never_reveal_labels[packet] is True]
    )


def _evaluate_order(
    exact: Sequence[ExactSearchResult],
    order: Sequence[int],
    levels: Sequence[float],
    oracle_tokens: Sequence[int],
) -> dict[str, Any]:
    chain = best_prefix_nested_chain(exact, order, levels)
    by_state = {row.state: row for row in exact}
    full_tokens = by_state[(1,) * len(order)].tokens
    anchors = []
    learned_tokens = 0
    all_success = True
    for level, predicted, baseline_tokens in zip(levels, chain, oracle_tokens):
        success = bool(predicted["feasible"])
        all_success &= success
        tokens = int(predicted["tokens"]) if predicted["tokens"] is not None else None
        if tokens is not None:
            learned_tokens += tokens
        anchors.append(
            {
                "fidelity_level": float(level),
                "contract_success": success,
                "tokens": tokens,
                "rate_regret_normalized_if_success": (
                    (tokens - baseline_tokens) / full_tokens
                    if success and tokens is not None
                    else None
                ),
            }
        )
    return {
        "all_active_contracts_success": all_success,
        "trajectory_regret_normalized": (
            (learned_tokens - sum(oracle_tokens)) / (len(levels) * full_tokens)
            if all_success
            else None
        ),
        "anchors": anchors,
    }


def _summarize_arm(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    anchors = [anchor for row in rows for anchor in row["anchors"]]
    regrets = [
        row["trajectory_regret_normalized"]
        for row in rows
        if row["trajectory_regret_normalized"] is not None
    ]
    levels = sorted({anchor["fidelity_level"] for anchor in anchors})
    return {
        "active_contract_success_fraction": mean(
            int(anchor["contract_success"]) for anchor in anchors
        ),
        "all_active_trajectory_success_fraction": mean(
            int(row["all_active_contracts_success"]) for row in rows
        ),
        "highest_active_anchor_success_fraction": mean(
            int(row["anchors"][-1]["contract_success"]) for row in rows
        ),
        "mean_normalized_ranking_regret_feasible": mean(regrets) if regrets else None,
        "per_level": {
            str(level): {
                "examples": len(values := [
                    anchor for anchor in anchors if anchor["fidelity_level"] == level
                ]),
                "successes": sum(anchor["contract_success"] for anchor in values),
                "contract_success_fraction": mean(
                    int(anchor["contract_success"]) for anchor in values
                ),
                "mean_rate_regret_normalized_successful": mean(
                    anchor["rate_regret_normalized_if_success"]
                    for anchor in values
                    if anchor["rate_regret_normalized_if_success"] is not None
                ),
            }
            for level in levels
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Counterfactual partial-order and never-reveal projections"
    )
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument(
        "--run", nargs=2, action="append", metavar=("LABEL", "DETAILS"), required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--required-contract-lower-bound", type=float, default=0.90)
    args = parser.parse_args()

    oracle_rows = {row["example_id"]: row for row in read_jsonl(args.oracle)}
    run_rows = {
        label: {row["example_id"]: row for row in read_jsonl(path)}
        for label, path in args.run
    }
    if len(run_rows) != len(args.run):
        raise ValueError("run labels must be unique")
    if any(set(rows) != set(oracle_rows) for rows in run_rows.values()):
        raise ValueError("every run must contain exactly the rank-oracle examples")

    arms = ("raw", "partial_order_projection", "never_reveal_tail_projection")
    evaluations = {
        label: {arm: [] for arm in arms}
        for label in run_rows
    }
    transitions = {
        label: {arm: Counter() for arm in arms[1:]}
        for label in run_rows
    }
    for example_id, source in oracle_rows.items():
        exact = list(
            read_jsonl(
                Path(args.exact_dir) / f"{example_id}.jsonl", ExactSearchResult
            )
        )
        for label, predictions in run_rows.items():
            prediction = predictions[example_id]
            raw_order = tuple(prediction["learned_order"])
            orders = {
                "raw": raw_order,
                "partial_order_projection": stable_partial_order_projection(
                    raw_order, source["pairwise_preferences"]
                ),
                "never_reveal_tail_projection": never_reveal_tail_projection(
                    raw_order, source["never_reveal_labels"]
                ),
            }
            levels = [float(value) for value in source["active_levels"]]
            oracle_tokens = [int(anchor["oracle_nested_tokens"]) for anchor in prediction["anchors"]]
            example_results = {
                arm: _evaluate_order(exact, order, levels, oracle_tokens)
                for arm, order in orders.items()
            }
            for arm, result in example_results.items():
                evaluations[label][arm].append(result)
            raw_success = example_results["raw"]["all_active_contracts_success"]
            for arm in arms[1:]:
                projected_success = example_results[arm]["all_active_contracts_success"]
                if projected_success and not raw_success:
                    transitions[label][arm]["rescued"] += 1
                elif raw_success and not projected_success:
                    transitions[label][arm]["broken"] += 1

    summaries = {
        label: {arm: _summarize_arm(rows) for arm, rows in by_arm.items()}
        for label, by_arm in evaluations.items()
    }
    active_levels = sorted(
        {
            float(level)
            for source in oracle_rows.values()
            for level in source["active_levels"]
        }
    )
    corrected_alpha = args.familywise_alpha / len(active_levels)
    for label, by_arm in summaries.items():
        for arm, summary in by_arm.items():
            for level, values in summary["per_level"].items():
                examples = values["examples"]
                successes = values["successes"]
                required = minimum_successes_for_lower_bound(
                    examples, args.required_contract_lower_bound, corrected_alpha
                )
                values["bonferroni_clopper_pearson_lower"] = clopper_pearson_lower(
                    successes, examples, corrected_alpha
                )
                values["required_successes_for_target_lower_bound"] = required
                values["additional_successes_needed"] = max(0, required - successes)

    output = {
        "complete": True,
        "status": "oracle_label_counterfactual_on_consumed_validation_not_deployable",
        "purpose": "isolate whether known partial-order or robust never-reveal violations explain learned-prefix failures",
        "does_not_modify_frozen_rank_only_gate": True,
        "examples": len(oracle_rows),
        "familywise_alpha": args.familywise_alpha,
        "per_anchor_alpha": corrected_alpha,
        "required_contract_lower_bound": args.required_contract_lower_bound,
        "runs": summaries,
        "trajectory_transitions": {
            label: {arm: dict(counts) for arm, counts in by_arm.items()}
            for label, by_arm in transitions.items()
        },
        "artifacts": {
            "oracle": args.oracle,
            "oracle_sha256": sha256(args.oracle),
            "runs": [
                {"label": label, "details": path, "details_sha256": sha256(path)}
                for label, path in args.run
            ],
        },
    }
    write_metadata(args.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
