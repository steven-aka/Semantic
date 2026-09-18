from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

from src.data.schemas import ExactSearchResult, read_jsonl
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.search.near_optimal_chain_set import (
    _min_subsets,
    _min_supersets,
    identifiable_pair_relations,
    near_optimal_chain_membership,
    summarize_chain_membership,
)


def memberships_for_slacks(
    exact: list[ExactSearchResult], levels: list[float], slacks: list[float]
) -> dict[float, dict[str, object]]:
    """Compute the existing exact chain DP once and threshold it at each slack.

    This is mechanically equivalent to independent calls to
    ``near_optimal_chain_membership``; the frozen objective is unchanged.
    """
    if not slacks or any(slack < 0 for slack in slacks):
        raise ValueError("slacks must be nonempty and nonnegative")
    width = len(exact[0].state)
    size = 1 << width
    by_mask = {state_to_mask(row.state): row for row in exact}
    if len(by_mask) != size:
        raise ValueError("complete binary lattice required")
    active = tuple(levels)
    first = [
        float(by_mask[mask].tokens) if by_mask[mask].fidelity >= active[0] else math.inf
        for mask in range(size)
    ]
    forward = [first]
    for level in active[1:]:
        prior = _min_subsets(forward[-1], width)
        forward.append([
            prior[mask] + by_mask[mask].tokens
            if by_mask[mask].fidelity >= level and not math.isinf(prior[mask])
            else math.inf
            for mask in range(size)
        ])
    backward = [[math.inf] * size for _ in active]
    backward[-1] = [
        float(by_mask[mask].tokens) if by_mask[mask].fidelity >= active[-1] else math.inf
        for mask in range(size)
    ]
    for index in range(len(active) - 2, -1, -1):
        later = _min_supersets(backward[index + 1], width)
        backward[index] = [
            later[mask] + by_mask[mask].tokens
            if by_mask[mask].fidelity >= active[index] and not math.isinf(later[mask])
            else math.inf
            for mask in range(size)
        ]
    optimum = min(forward[-1])
    normalizer = len(active) * by_mask[size - 1].tokens
    complete_costs = [
        [forward[index][mask] + backward[index][mask] - by_mask[mask].tokens for mask in range(size)]
        for index in range(len(active))
    ]
    return {
        slack: {
            "active_levels": list(active),
            "state_masks_by_level": [
                [mask for mask, cost in enumerate(costs) if cost <= optimum + slack * normalizer + 1e-9]
                for costs in complete_costs
            ],
        }
        for slack in slacks
    }


def graph_has_cycle(width: int, edges: set[tuple[int, int]]) -> bool:
    successors = [set() for _ in range(width)]
    indegree = [0] * width
    for winner, loser in edges:
        if loser not in successors[winner]:
            successors[winner].add(loser)
            indegree[loser] += 1
    queue = [index for index, degree in enumerate(indegree) if degree == 0]
    visited = 0
    while queue:
        winner = queue.pop()
        visited += 1
        for loser in successors[winner]:
            indegree[loser] -= 1
            if indegree[loser] == 0:
                queue.append(loser)
    return visited != width


def local_boundary_edges(
    fidelity: dict[int, float], masks_by_level: list[list[int]], levels: list[float], width: int
) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for level, masks in zip(levels, masks_by_level):
        for mask in masks:
            critical = [
                packet for packet in range(width)
                if mask & (1 << packet) and fidelity[mask ^ (1 << packet)] < level
            ]
            harmful = [
                packet for packet in range(width)
                if not mask & (1 << packet) and fidelity[mask | (1 << packet)] < level
            ]
            edges.update((winner, loser) for winner in critical for loser in harmful)
    return edges


def harmfulness_stability(
    fidelity: dict[int, float], masks_by_level: list[list[int]], levels: list[float], width: int
) -> dict[str, int]:
    """Compare a local harmful witness only to admissible states at its own anchor.

    Raw marginal sign and contract-crossing status are reported separately.
    No claim about the existence of a feasible static order follows from either.
    """
    counts: Counter[str] = Counter()
    for level, masks in zip(levels, masks_by_level):
        for packet in range(width):
            flag = 1 << packet
            absent = [mask for mask in masks if not mask & flag]
            if not absent:
                continue
            deltas = [fidelity[mask | flag] - fidelity[mask] for mask in absent]
            crossing = [fidelity[mask | flag] < level for mask in absent]
            if not any(crossing):
                continue
            counts["harmful_packet_anchor_groups"] += 1
            counts["harmful_witness_states"] += sum(crossing)
            counts["admissible_absent_states"] += len(absent)
            counts["contract_crossing_flip_groups"] += int(any(crossing) and not all(crossing))
            counts["strict_sign_flip_groups"] += int(min(deltas) < 0 < max(deltas))
            counts["negative_to_nonnegative_groups"] += int(min(deltas) < 0 <= max(deltas))
            counts["nonnegative_harmful_witness_groups"] += int(
                any(delta >= 0 and harmful for delta, harmful in zip(deltas, crossing))
            )
    return dict(counts)


def summarize_prediction(
    predictions: dict[str, dict], boundary_rows: dict[str, dict]
) -> dict[str, float | int]:
    correct = pairs = supervised = complete = trajectories = 0
    for example_id, prediction in predictions.items():
        preferences = boundary_rows[example_id]["pairwise_preferences"]
        scores = prediction["scores"]
        correct += sum(scores[winner] > scores[loser] for winner, loser in preferences)
        pairs += len(preferences)
        if preferences:
            supervised += 1
            complete += int(all(scores[winner] > scores[loser] for winner, loser in preferences))
        trajectories += int(prediction["all_active_contracts_success"])
    return {
        "examples": len(predictions),
        "boundary_pairs": pairs,
        "boundary_pair_accuracy": correct / pairs if pairs else None,
        "supervised_examples": supervised,
        "complete_boundary_separation": complete / supervised if supervised else None,
        "oracle_cutoff_all_active_trajectory_success": trajectories / len(predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only V3 model-assumption audit on consumed roles")
    parser.add_argument("--split", nargs=3, action="append", metavar=("NAME", "ORACLE", "BOUNDARY"), required=True)
    parser.add_argument("--prediction", nargs=2, action="append", metavar=("NAME", "DETAILS"), default=[])
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--slacks", nargs="+", type=float, default=[0.0, 0.005, 0.01, 0.05])
    args = parser.parse_args()
    allowed_consumed_roles = {"train2000", "development300", "confirm210"}
    if any(name not in allowed_consumed_roles for name, _, _ in args.split):
        raise ValueError("only the consumed train/development/confirmation roles are permitted")
    if any(name not in allowed_consumed_roles for name, _ in args.prediction):
        raise ValueError("prediction role is not a consumed role")
    forbidden_path_tokens = ("calibration300", "final_test300", "test300")
    if any(
        token in path.lower()
        for _, oracle_path, boundary_path in args.split
        for path in (oracle_path, boundary_path)
        for token in forbidden_path_tokens
    ) or any(
        token in path.lower()
        for _, path in args.prediction
        for token in forbidden_path_tokens
    ):
        raise ValueError("locked calibration/final-test roles must not enter this audit")
    if len({name for name, _, _ in args.split}) != len(args.split):
        raise ValueError("split names must be unique")
    predictions = {name: {row["example_id"]: row for row in read_jsonl(path)} for name, path in args.prediction}
    if len(predictions) != len(args.prediction):
        raise ValueError("prediction names must be unique")
    output: dict[str, object] = {
        "status": "post_hoc_consumed_split_diagnostic_not_model_selection",
        "slack_definition": "extra cumulative tokens / (active anchor count * full-state tokens)",
        "slacks": args.slacks,
        "splits": {},
        "locked_roles_evaluated": False,
    }
    for name, oracle_path, boundary_path in args.split:
        oracle = {row["example_id"]: row for row in read_jsonl(oracle_path)}
        boundaries = {row["example_id"]: row for row in read_jsonl(boundary_path)}
        if set(oracle) != set(boundaries):
            raise ValueError(f"oracle/boundary IDs differ in {name}")
        if name in predictions and set(predictions[name]) != set(oracle):
            raise ValueError(f"prediction IDs differ in {name}")
        aggregate: Counter[str] = Counter()
        slack_rows: dict[float, list[dict]] = {slack: [] for slack in args.slacks}
        for row_index, (example_id, source) in enumerate(oracle.items(), 1):
            exact = list(read_jsonl(Path(args.exact_dir) / f"{example_id}.jsonl", ExactSearchResult))
            width = len(source["packet_ids"])
            fidelity = {state_to_mask(row.state): float(row.fidelity) for row in exact}
            if len(fidelity) != 1 << width:
                raise ValueError(f"incomplete lattice: {example_id}")
            all_memberships = memberships_for_slacks(exact, source["active_levels"], args.slacks)
            for slack in args.slacks:
                membership = all_memberships[slack]
                masks = membership["state_masks_by_level"]
                levels = membership["active_levels"]
                relation = identifiable_pair_relations(membership, width)
                stable = {
                    (first, second) if direction == -1 else (second, first)
                    for (first, second), direction in relation.items()
                }
                local = local_boundary_edges(fidelity, masks, levels, width)
                retained = local & stable
                ambiguity = summarize_chain_membership(membership, width)
                slack_rows[slack].append({
                    "packet_anchor_ambiguity": ambiguity["packet_anchor_ambiguity_fraction"],
                    "local_edges": len(local),
                    "stable_edges": len(stable),
                    "stable_boundary_edges": len(retained),
                    "raw_cycle": graph_has_cycle(width, local),
                    "raw_direct_conflict": any((loser, winner) in local for winner, loser in local),
                    "stable_cycle": graph_has_cycle(width, stable),
                })
                if slack == 0.005:
                    aggregate.update(harmfulness_stability(fidelity, masks, levels, width))
                    if retained != {tuple(pair) for pair in boundaries[example_id]["pairwise_preferences"]}:
                        raise ValueError(f"frozen V3 stable labels disagree: {example_id}")
            if row_index % 50 == 0 or row_index == len(oracle):
                print(json.dumps({"split": name, "processed": row_index, "total": len(oracle)}), flush=True)
        harmful_groups = aggregate["harmful_packet_anchor_groups"]
        split_summary = {
            "examples": len(oracle),
            "harmfulness_at_frozen_slack_0_005": dict(aggregate),
            "harmfulness_rates_at_frozen_slack_0_005": {
                key: aggregate[key] / harmful_groups if harmful_groups else None
                for key in (
                    "strict_sign_flip_groups",
                    "negative_to_nonnegative_groups",
                    "contract_crossing_flip_groups",
                    "nonnegative_harmful_witness_groups",
                )
            },
            "slack_sensitivity": {
                str(slack): {
                    "mean_packet_anchor_ambiguity": mean(row["packet_anchor_ambiguity"] for row in rows),
                    "mean_raw_local_edges": mean(row["local_edges"] for row in rows),
                    "mean_stable_boundary_edges": mean(row["stable_boundary_edges"] for row in rows),
                    "fraction_with_raw_local_cycle": mean(row["raw_cycle"] for row in rows),
                    "fraction_with_direct_local_conflict": mean(row["raw_direct_conflict"] for row in rows),
                    "fraction_with_stable_relation_cycle": mean(row["stable_cycle"] for row in rows),
                }
                for slack, rows in slack_rows.items()
            },
            "selected_v3_prediction": summarize_prediction(predictions[name], boundaries) if name in predictions else None,
            "oracle_sha256": sha256(oracle_path),
            "boundary_sha256": sha256(boundary_path),
        }
        output["splits"][name] = split_summary
        print(json.dumps({"completed_split": name, "examples": len(oracle)}), flush=True)
    output["prediction_sha256"] = {name: sha256(path) for name, path in args.prediction}
    write_metadata(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
