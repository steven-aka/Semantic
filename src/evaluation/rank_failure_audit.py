from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from scipy.stats import beta

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata
from src.search.rank_then_cut import prefix_state, validate_packet_order


def clopper_pearson_lower(successes: int, examples: int, alpha: float) -> float:
    if not 0 <= successes <= examples or examples <= 0:
        raise ValueError("successes must be between zero and a positive example count")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, examples - successes + 1))


def minimum_successes_for_lower_bound(
    examples: int, required_lower_bound: float, alpha: float
) -> int:
    if not 0 < required_lower_bound < 1:
        raise ValueError("required lower bound must be in (0, 1)")
    return next(
        successes
        for successes in range(examples + 1)
        if clopper_pearson_lower(successes, examples, alpha) >= required_lower_bound
    )


def _state_from_set(selected: set[int], width: int) -> tuple[int, ...]:
    return tuple(int(index in selected) for index in range(width))


def _packet_record(source: dict[str, Any], scores: Sequence[float], index: int) -> dict[str, Any]:
    return {
        "packet_index": index,
        "score": float(scores[index]),
        "never_reveal_label": source["never_reveal_labels"][index],
        "text": source["packet_texts"][index],
    }


def audit_failed_order(
    source: dict[str, Any],
    exact: Sequence[ExactSearchResult],
    prediction: dict[str, Any],
    *,
    run_label: str,
) -> dict[str, Any]:
    """Find the smallest exact-state repair for one failed learned order.

    Adjacent swaps are the cross-boundary inversions required to stably move
    all packets in a feasible state before all packets outside that state.
    This is a diagnostic distance, not a learned inference-time operation.
    """
    if prediction["all_active_contracts_success"]:
        raise ValueError("failure audit requires a failed trajectory")
    width = len(source["packet_ids"])
    order = validate_packet_order(prediction["learned_order"], width)
    scores = [float(value) for value in prediction["scores"]]
    if len(scores) != width:
        raise ValueError("score count does not match packet width")
    by_state = {row.state: row for row in exact}
    if len(by_state) != 1 << width:
        raise ValueError("failure audit requires a complete binary exact lattice")
    target = float(source["active_levels"][-1])
    positions = {packet: position for position, packet in enumerate(order)}
    all_packets = set(range(width))

    best_prefix_count, best_prefix_row = max(
        (
            (count, by_state[prefix_state(order, count)])
            for count in range(width + 1)
        ),
        key=lambda item: (item[1].fidelity, -item[1].tokens, -item[0]),
    )

    repairs = []
    for row in exact:
        if row.fidelity < target:
            continue
        selected = {index for index, value in enumerate(row.state) if value}
        excluded = all_packets - selected
        count = len(selected)
        current_prefix = set(order[:count])
        missing = selected - current_prefix
        intruders = current_prefix - selected
        adjacent_swaps = sum(
            positions[excluded_packet] < positions[selected_packet]
            for selected_packet in selected
            for excluded_packet in excluded
        )
        repairs.append(
            (
                (
                    adjacent_swaps,
                    len(missing),
                    row.tokens,
                    -row.fidelity,
                    row.state,
                ),
                row,
                missing,
                intruders,
            )
        )
    if not repairs:
        raise ValueError("highest active anchor has no feasible exact state")
    repair_key, repaired_row, missing, intruders = min(repairs, key=lambda item: item[0])

    preferences = {tuple(pair) for pair in source["pairwise_preferences"]}
    relation_counts: Counter[str] = Counter()
    relation_rows = []
    for missing_packet in sorted(missing):
        for intruder in sorted(intruders):
            if (missing_packet, intruder) in preferences:
                relation = "identifiable_model_violation"
            elif (intruder, missing_packet) in preferences:
                relation = "supervision_opposes_repair"
            else:
                relation = "unlabeled"
            relation_counts[relation] += 1
            relation_rows.append(
                {
                    "missing_packet_index": missing_packet,
                    "intruder_packet_index": intruder,
                    "relation": relation,
                }
            )

    one_add = one_drop = one_swap = False
    for count in range(width + 1):
        current = set(order[:count])
        excluded = all_packets - current
        one_add |= any(
            by_state[_state_from_set(current | {packet}, width)].fidelity >= target
            for packet in excluded
        )
        one_drop |= any(
            by_state[_state_from_set(current - {packet}, width)].fidelity >= target
            for packet in current
        )
        one_swap |= any(
            by_state[_state_from_set((current - {old}) | {new}, width)].fidelity >= target
            for old in current
            for new in excluded
        )

    return {
        "run_label": run_label,
        "example_id": source["example_id"],
        "question": source["question"],
        "highest_active_fidelity": target,
        "learned_order": list(order),
        "best_learned_prefix_count": best_prefix_count,
        "best_learned_prefix_fidelity": best_prefix_row.fidelity,
        "fidelity_shortfall": target - best_prefix_row.fidelity,
        "minimum_adjacent_swaps_to_a_feasible_state": repair_key[0],
        "minimum_boundary_replacements": repair_key[1],
        "closest_feasible_state": list(repaired_row.state),
        "closest_feasible_state_tokens": repaired_row.tokens,
        "closest_feasible_state_fidelity": repaired_row.fidelity,
        "missing_packets": [
            _packet_record(source, scores, index) for index in sorted(missing)
        ],
        "intruder_packets": [
            _packet_record(source, scores, index) for index in sorted(intruders)
        ],
        "repair_pair_relations": relation_rows,
        "repair_pair_relation_counts": dict(relation_counts),
        "one_edit_counterfactual": {
            "add": one_add,
            "drop": one_drop,
            "swap": one_swap,
        },
        "packet_anchor_ambiguity_fraction": source[
            "packet_anchor_ambiguity_fraction"
        ],
        "identifiable_pairs": source["identifiable_pairs"],
    }


def summarize_audit(
    audit_rows: Sequence[dict[str, Any]],
    run_details: dict[str, Sequence[dict[str, Any]]],
    *,
    familywise_alpha: float,
    required_contract_lower_bound: float,
) -> dict[str, Any]:
    run_labels = list(run_details)
    failed_by_run = {
        label: {
            row["example_id"]
            for row in rows
            if not row["all_active_contracts_success"]
        }
        for label, rows in run_details.items()
    }
    common_failures = set.intersection(*failed_by_run.values())
    union_failures = set.union(*failed_by_run.values())
    relation_counts: Counter[str] = Counter()
    for row in audit_rows:
        relation_counts.update(row["repair_pair_relation_counts"])
    relation_total = sum(relation_counts.values())

    active_levels = sorted(
        {
            float(anchor["fidelity_level"])
            for rows in run_details.values()
            for row in rows
            for anchor in row["anchors"]
        }
    )
    corrected_alpha = familywise_alpha / len(active_levels)
    risk_rows = []
    for label, rows in run_details.items():
        for level in active_levels:
            anchors = [
                anchor
                for row in rows
                for anchor in row["anchors"]
                if float(anchor["fidelity_level"]) == level
            ]
            if not anchors:
                continue
            successes = sum(bool(anchor["contract_success"]) for anchor in anchors)
            examples = len(anchors)
            required = minimum_successes_for_lower_bound(
                examples, required_contract_lower_bound, corrected_alpha
            )
            risk_rows.append(
                {
                    "run_label": label,
                    "fidelity_level": level,
                    "successes": successes,
                    "examples": examples,
                    "success_fraction": successes / examples,
                    "bonferroni_clopper_pearson_lower": clopper_pearson_lower(
                        successes, examples, corrected_alpha
                    ),
                    "required_successes_for_target_lower_bound": required,
                    "additional_successes_needed": max(0, required - successes),
                }
            )

    return {
        "complete": True,
        "status": "post_hoc_validation_diagnostic_not_a_new_gate",
        "does_not_modify_frozen_rank_only_gate": True,
        "runs": run_labels,
        "evaluated_run_trajectories": sum(len(rows) for rows in run_details.values()),
        "failed_run_trajectories": len(audit_rows),
        "unique_failed_examples": len(union_failures),
        "common_failed_examples": len(common_failures),
        "common_failed_example_ids": sorted(common_failures),
        "repair": {
            "minimum_boundary_replacements": dict(
                sorted(Counter(row["minimum_boundary_replacements"] for row in audit_rows).items())
            ),
            "minimum_adjacent_swaps": dict(
                sorted(
                    Counter(
                        row["minimum_adjacent_swaps_to_a_feasible_state"]
                        for row in audit_rows
                    ).items()
                )
            ),
            "one_edit_counterfactual_counts": {
                operation: sum(
                    row["one_edit_counterfactual"][operation] for row in audit_rows
                )
                for operation in ("add", "drop", "swap")
            },
            "repair_pair_relation_counts": dict(relation_counts),
            "identifiable_model_violation_fraction": (
                relation_counts["identifiable_model_violation"] / relation_total
                if relation_total
                else None
            ),
            "intruder_packets": sum(len(row["intruder_packets"]) for row in audit_rows),
            "robust_never_reveal_intruder_packets": sum(
                packet["never_reveal_label"] is True
                for row in audit_rows
                for packet in row["intruder_packets"]
            ),
            "mean_fidelity_shortfall": mean(
                row["fidelity_shortfall"] for row in audit_rows
            ),
        },
        "risk_headroom": {
            "familywise_alpha": familywise_alpha,
            "bonferroni_families": len(active_levels),
            "per_anchor_alpha": corrected_alpha,
            "required_contract_lower_bound": required_contract_lower_bound,
            "rows": risk_rows,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit exact prefix repairs for failed learned packet orders"
    )
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        metavar=("LABEL", "DETAILS"),
        required=True,
    )
    parser.add_argument("--details-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--required-contract-lower-bound", type=float, default=0.90)
    args = parser.parse_args()

    oracle_rows = {row["example_id"]: row for row in read_jsonl(args.oracle)}
    run_details: dict[str, list[dict[str, Any]]] = {}
    artifacts = {"oracle": args.oracle, "oracle_sha256": sha256(args.oracle), "runs": []}
    audit_rows = []
    expected_ids: set[str] | None = None
    for label, path in args.run:
        if label in run_details:
            raise ValueError(f"duplicate run label: {label}")
        rows = list(read_jsonl(path))
        ids = {row["example_id"] for row in rows}
        if expected_ids is not None and ids != expected_ids:
            raise ValueError("run detail files contain different example IDs")
        expected_ids = ids
        if ids != set(oracle_rows):
            raise ValueError("run details and rank oracle contain different example IDs")
        run_details[label] = rows
        artifacts["runs"].append({"label": label, "details": path, "sha256": sha256(path)})
        for prediction in rows:
            if prediction["all_active_contracts_success"]:
                continue
            example_id = prediction["example_id"]
            exact = list(
                read_jsonl(
                    Path(args.exact_dir) / f"{example_id}.jsonl", ExactSearchResult
                )
            )
            audit_rows.append(
                audit_failed_order(
                    oracle_rows[example_id], exact, prediction, run_label=label
                )
            )

    summary = summarize_audit(
        audit_rows,
        run_details,
        familywise_alpha=args.familywise_alpha,
        required_contract_lower_bound=args.required_contract_lower_bound,
    )
    summary["artifacts"] = artifacts
    write_jsonl(args.details_output, audit_rows)
    write_metadata(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
