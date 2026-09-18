from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.evaluation.rank_failure_audit import (
    clopper_pearson_lower,
    minimum_successes_for_lower_bound,
)
from src.evaluation.rank_projection_diagnostic import (
    _evaluate_order,
    _summarize_arm,
    stable_partial_order_projection,
)
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.search.near_optimal_chain_set import near_optimal_chain_membership


def derive_critical_boundary_preferences(
    fidelity_by_mask: Mapping[int, float],
    state_masks_by_level: Sequence[Sequence[int]],
    levels: Sequence[float],
    width: int,
    identifiable_preferences: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Derive conservative boundaries from every admissible oracle state.

    ``robust_membership`` separates packets included in every admissible state
    at an anchor from packets excluded from every state at that anchor.
    ``stable_counterfactual`` retains only add/drop critical--harmful pairs
    whose direction is also identifiable across the complete near-optimal
    chain set. This avoids assigning an ambiguous relation from one arbitrarily
    selected oracle state.
    """
    if len(state_masks_by_level) != len(levels):
        raise ValueError("state memberships and levels must align")
    if set(fidelity_by_mask) != set(range(1 << width)):
        raise ValueError("complete binary fidelity lattice required")

    identifiable = {
        (int(pair[0]), int(pair[1])) for pair in identifiable_preferences
    }
    robust_membership: set[tuple[int, int]] = set()
    local_counterfactual: Counter[tuple[int, int]] = Counter()
    local_directions: set[tuple[int, int]] = set()
    critical_witnesses = harmful_witnesses = state_witnesses = 0

    for level, raw_masks in zip(levels, state_masks_by_level):
        masks = tuple(int(mask) for mask in raw_masks)
        if not masks:
            raise ValueError("every active anchor needs admissible states")
        if any(fidelity_by_mask[mask] < level for mask in masks):
            raise ValueError("admissible state violates its fidelity anchor")

        robust_included = [
            packet
            for packet in range(width)
            if all(mask & (1 << packet) for mask in masks)
        ]
        robust_excluded = [
            packet
            for packet in range(width)
            if all(not mask & (1 << packet) for mask in masks)
        ]
        robust_membership.update(
            (included, excluded)
            for included in robust_included
            for excluded in robust_excluded
        )

        for mask in masks:
            critical = [
                packet
                for packet in range(width)
                if mask & (1 << packet)
                and fidelity_by_mask[mask ^ (1 << packet)] < level
            ]
            harmful = [
                packet
                for packet in range(width)
                if not mask & (1 << packet)
                and fidelity_by_mask[mask | (1 << packet)] < level
            ]
            state_witnesses += 1
            critical_witnesses += len(critical)
            harmful_witnesses += len(harmful)
            for winner in critical:
                for loser in harmful:
                    local_counterfactual[(winner, loser)] += 1
                    local_directions.add((winner, loser))

    conflicting_unordered_pairs = {
        tuple(sorted((winner, loser)))
        for winner, loser in local_directions
        if (loser, winner) in local_directions
    }
    stable_counterfactual = set(local_counterfactual) & identifiable
    robust_membership &= identifiable
    return {
        "robust_membership": sorted(robust_membership),
        "stable_counterfactual": sorted(stable_counterfactual),
        "all_identifiable": sorted(identifiable),
        "local_counterfactual_directed_pairs": len(local_counterfactual),
        "conflicting_local_unordered_pairs": len(conflicting_unordered_pairs),
        "admissible_state_witnesses": state_witnesses,
        "critical_packet_witnesses": critical_witnesses,
        "harmful_packet_witnesses": harmful_witnesses,
    }


def linear_extension_contract_statistics(
    fidelity_by_mask: Mapping[int, float],
    width: int,
    preferences: Sequence[Sequence[int]],
    target: float,
) -> dict[str, Any]:
    """Count exact linear extensions whose prefix path reaches ``target``."""
    predecessors = [0] * width
    seen = set()
    for raw_pair in preferences:
        winner, loser = (int(value) for value in raw_pair)
        if not 0 <= winner < width or not 0 <= loser < width or winner == loser:
            raise ValueError("invalid boundary preference")
        if (winner, loser) in seen:
            continue
        seen.add((winner, loser))
        predecessors[loser] |= 1 << winner

    size = 1 << width
    total = [0] * size
    successful = [0] * size
    total[0] = 1
    successful[0] = int(fidelity_by_mask[0] >= target)
    feasible_ideals = 0
    for mask in range(size):
        if total[mask] and fidelity_by_mask[mask] >= target:
            feasible_ideals += 1
        for packet in range(width):
            flag = 1 << packet
            if mask & flag or predecessors[packet] & ~mask:
                continue
            next_mask = mask | flag
            total[next_mask] += total[mask]
            successful[next_mask] += successful[mask]
            if fidelity_by_mask[next_mask] >= target:
                successful[next_mask] += total[mask] - successful[mask]

    final = size - 1
    if total[final] == 0:
        raise ValueError("boundary preferences contain a cycle")
    return {
        "linear_extensions": total[final],
        "successful_linear_extensions": successful[final],
        "successful_linear_extension_fraction": successful[final] / total[final],
        "feasible_ideal_states": feasible_ideals,
        "at_least_one_successful_extension": bool(successful[final]),
        "all_extensions_successful": successful[final] == total[final],
    }


def _add_risk_bounds(
    summary: dict[str, Any],
    *,
    corrected_alpha: float,
    required_contract_lower_bound: float,
) -> None:
    for values in summary["per_level"].values():
        examples = int(values["examples"])
        successes = int(values["successes"])
        values["bonferroni_clopper_pearson_lower"] = clopper_pearson_lower(
            successes, examples, corrected_alpha
        )
        required = minimum_successes_for_lower_bound(
            examples, required_contract_lower_bound, corrected_alpha
        )
        values["required_successes_for_target_lower_bound"] = required
        values["additional_successes_needed"] = max(0, required - successes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set-valued exhaustive critical-prefix boundary diagnostic"
    )
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument(
        "--run", nargs=2, action="append", metavar=("LABEL", "DETAILS"), required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--details", required=True)
    parser.add_argument("--normalized-slack", type=float, default=0.005)
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
        raise ValueError("every run must contain exactly the oracle examples")

    relation_arms = (
        "robust_membership",
        "stable_counterfactual",
        "all_identifiable",
    )
    evaluation_arms = ("raw",) + tuple(f"{name}_projection" for name in relation_arms)
    evaluations = {
        label: {arm: [] for arm in evaluation_arms} for label in run_rows
    }
    transitions = {
        label: {arm: Counter() for arm in evaluation_arms[1:]}
        for label in run_rows
    }
    details = []

    for example_id, source in oracle_rows.items():
        exact = list(
            read_jsonl(Path(args.exact_dir) / f"{example_id}.jsonl", ExactSearchResult)
        )
        width = len(source["packet_ids"])
        fidelity_by_mask = {state_to_mask(row.state): float(row.fidelity) for row in exact}
        membership = near_optimal_chain_membership(
            exact,
            [float(value) for value in source["active_levels"]],
            normalized_slack=args.normalized_slack,
        )
        boundaries = derive_critical_boundary_preferences(
            fidelity_by_mask,
            membership["state_masks_by_level"],
            membership["active_levels"],
            width,
            source["pairwise_preferences"],
        )
        target = float(source["active_levels"][-1])
        extension_stats = {
            arm: linear_extension_contract_statistics(
                fidelity_by_mask, width, boundaries[arm], target
            )
            for arm in relation_arms
        }
        detail = {
            "example_id": example_id,
            "highest_active_fidelity": target,
            "relation_counts": {arm: len(boundaries[arm]) for arm in relation_arms},
            "local_counterfactual_directed_pairs": boundaries[
                "local_counterfactual_directed_pairs"
            ],
            "conflicting_local_unordered_pairs": boundaries[
                "conflicting_local_unordered_pairs"
            ],
            "admissible_state_witnesses": boundaries["admissible_state_witnesses"],
            "critical_packet_witnesses": boundaries["critical_packet_witnesses"],
            "harmful_packet_witnesses": boundaries["harmful_packet_witnesses"],
            "linear_extension_contract": extension_stats,
            "runs": {},
        }

        for label, predictions in run_rows.items():
            prediction = predictions[example_id]
            raw_order = tuple(int(value) for value in prediction["learned_order"])
            orders = {"raw": raw_order}
            orders.update(
                {
                    f"{arm}_projection": stable_partial_order_projection(
                        raw_order, boundaries[arm]
                    )
                    for arm in relation_arms
                }
            )
            levels = [float(value) for value in source["active_levels"]]
            oracle_tokens = [
                int(anchor["oracle_nested_tokens"]) for anchor in prediction["anchors"]
            ]
            results = {
                arm: _evaluate_order(exact, order, levels, oracle_tokens)
                for arm, order in orders.items()
            }
            for arm, result in results.items():
                evaluations[label][arm].append(result)
            raw_success = results["raw"]["all_active_contracts_success"]
            for arm in evaluation_arms[1:]:
                projected_success = results[arm]["all_active_contracts_success"]
                if projected_success and not raw_success:
                    transitions[label][arm]["rescued"] += 1
                elif raw_success and not projected_success:
                    transitions[label][arm]["broken"] += 1
                else:
                    transitions[label][arm][
                        "unchanged_success" if raw_success else "unchanged_failure"
                    ] += 1
            detail["runs"][label] = {
                arm: {
                    "all_active_contracts_success": result[
                        "all_active_contracts_success"
                    ],
                    "highest_active_contract_success": result["anchors"][-1][
                        "contract_success"
                    ],
                }
                for arm, result in results.items()
            }
        details.append(detail)

    active_levels = sorted(
        {
            float(value)
            for source in oracle_rows.values()
            for value in source["active_levels"]
        }
    )
    corrected_alpha = args.familywise_alpha / len(active_levels)
    run_summaries = {
        label: {arm: _summarize_arm(rows) for arm, rows in by_arm.items()}
        for label, by_arm in evaluations.items()
    }
    for by_arm in run_summaries.values():
        for summary in by_arm.values():
            _add_risk_bounds(
                summary,
                corrected_alpha=corrected_alpha,
                required_contract_lower_bound=args.required_contract_lower_bound,
            )

    relation_summary = {}
    for arm in relation_arms:
        counts = [row["relation_counts"][arm] for row in details]
        extension_fractions = [
            row["linear_extension_contract"][arm][
                "successful_linear_extension_fraction"
            ]
            for row in details
        ]
        relation_summary[arm] = {
            "mean_relations_per_example": mean(counts),
            "examples_with_no_relations": sum(count == 0 for count in counts),
            "mean_successful_linear_extension_fraction": mean(extension_fractions),
            "examples_with_any_successful_extension": sum(
                row["linear_extension_contract"][arm][
                    "at_least_one_successful_extension"
                ]
                for row in details
            ),
            "examples_where_all_extensions_succeed": sum(
                row["linear_extension_contract"][arm]["all_extensions_successful"]
                for row in details
            ),
        }

    output = {
        "complete": True,
        "status": "consumed_validation_diagnostic_not_model_selection_or_heldout",
        "purpose": "test whether set-valued critical-boundary supervision is sufficient before authorizing another training objective",
        "does_not_modify_frozen_v2_or_v2_1_decisions": True,
        "examples": len(oracle_rows),
        "normalized_slack": args.normalized_slack,
        "familywise_alpha": args.familywise_alpha,
        "per_anchor_alpha": corrected_alpha,
        "required_contract_lower_bound": args.required_contract_lower_bound,
        "relation_summary": relation_summary,
        "local_counterfactual_conflict": {
            "examples_with_conflicting_pair_directions": sum(
                row["conflicting_local_unordered_pairs"] > 0 for row in details
            ),
            "total_conflicting_unordered_pairs": sum(
                row["conflicting_local_unordered_pairs"] for row in details
            ),
        },
        "runs": run_summaries,
        "trajectory_transitions": {
            label: {arm: dict(values) for arm, values in by_arm.items()}
            for label, by_arm in transitions.items()
        },
        "artifacts": {
            "oracle": args.oracle,
            "oracle_sha256": sha256(args.oracle),
            "details": args.details,
            "runs": [
                {"label": label, "details": path, "details_sha256": sha256(path)}
                for label, path in args.run
            ],
        },
    }
    write_jsonl(args.details, details)
    write_metadata(args.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
