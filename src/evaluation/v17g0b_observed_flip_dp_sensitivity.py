from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.training.train_v17b1_multi_anchor_viability import read_gzip


def reachability(success: list[bool], width: int) -> list[bool]:
    result = success[:]
    for mask in range((1 << width) - 1, -1, -1):
        if not result[mask]:
            result[mask] = any(not mask & (1 << packet) and result[mask | (1 << packet)] for packet in range(width))
    return result


def strict_pairs(reach: list[bool], row: dict) -> set[tuple[int, int]]:
    mask = int(row["selected_mask"])
    positive = [int(action["packet"]) for action in row["actions"] if reach[mask | (1 << int(action["packet"]))]]
    negative = [int(action["packet"]) for action in row["actions"] if not reach[mask | (1 << int(action["packet"]))]]
    return {(plus, minus) for plus in positive for minus in negative}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", required=True)
    parser.add_argument("--g0a-results", required=True)
    parser.add_argument("--training-artifact", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.protocol_config).read_text())
    if config["status"] != "FROZEN_APPROVED_TO_RUN":
        raise ValueError("unfrozen protocol")
    observations = defaultdict(list)
    for row in read_jsonl(args.g0a_results):
        if any(item["success"] != row["cached_success"] for item in row["reruns"]):
            observations[row["example_id"]].append(row)
    train = defaultdict(list)
    for row in read_gzip(args.training_artifact):
        if row["example_id"] in observations and .9 in row["critical_levels"]:
            train[row["example_id"]].append(row)
    rows = []
    for query, changed in sorted(observations.items()):
        exact = list(read_jsonl(Path(args.exact_dir) / f"{query}.jsonl", ExactSearchResult))
        width = len(exact[0].state)
        success = [False] * (1 << width)
        for item in exact:
            success[state_to_mask(item.state)] = item.fidelity + 1e-12 >= .9
        baseline = reachability(success, width)
        query_states = train[query]
        for repeat in range(len(changed[0]["reruns"])):
            trial_success = success[:]
            for item in changed:
                trial_success[int(item["mask"])] = bool(item["reruns"][repeat]["success"])
            trial = reachability(trial_success, width)
            action_flips = deployed_action_flips = pair_changes = strict_states = 0
            for state in query_states:
                mask = int(state["selected_mask"])
                old = [baseline[mask | (1 << int(action["packet"]))] for action in state["actions"]]
                for action, expected in zip(state["actions"], old):
                    label = action["future_viability"][3]
                    if label is not None and bool(label) != expected:
                        raise ValueError("cached DP reachability does not reproduce the audited training label")
                new = [trial[mask | (1 << int(action["packet"]))] for action in state["actions"]]
                count = sum(a != b for a, b in zip(old, new))
                action_flips += count
                if state["provenance"] == "deployed":
                    deployed_action_flips += count
                old_pairs = strict_pairs(baseline, state)
                new_pairs = strict_pairs(trial, state)
                pair_changes += len(old_pairs ^ new_pairs)
                strict_states += bool(old_pairs != new_pairs)
            rows.append({"example_id": query, "repeat": repeat, "observed_flip_masks": sum(trial_success[int(item["mask"])] != success[int(item["mask"])] for item in changed), "all_mask_reachability_flips": sum(left != right for left, right in zip(baseline, trial)), "critical_states": len(query_states), "critical_state_action_target_flips": action_flips, "deployed_critical_state_action_target_flips": deployed_action_flips, "strict_pair_symmetric_difference": pair_changes, "strict_states_changed": strict_states})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "per_query_repeat.jsonl", rows)
    summary = {"complete": True, "queries": len(observations), "scenarios": len(rows), "scenarios_with_reachability_flip": sum(row["all_mask_reachability_flips"] > 0 for row in rows), "scenarios_with_critical_action_flip": sum(row["critical_state_action_target_flips"] > 0 for row in rows), "scenarios_with_strict_pair_change": sum(row["strict_pair_symmetric_difference"] > 0 for row in rows), "distinct_queries_with_critical_action_flip": len({row["example_id"] for row in rows if row["critical_state_action_target_flips"] > 0}), "sum_of_per_query_max_critical_action_flips": sum(max(row["critical_state_action_target_flips"] for row in rows if row["example_id"] == query) for query in observations), "sum_of_per_query_max_deployed_action_flips": sum(max(row["deployed_critical_state_action_target_flips"] for row in rows if row["example_id"] == query) for query in observations), "total_critical_action_flips_across_repeated_scenarios": sum(row["critical_state_action_target_flips"] for row in rows), "interpretation": config["interpretation"], "decision": "AUDIT_TARGET_REPRODUCIBILITY_BEFORE_SUPERVISION_REDESIGN", "training_authorized": False, "internal_used": False, "development_used": False, "confirmation_used": False, "artifacts": {"protocol_sha256": sha256(args.protocol_config), "g0a_results_sha256": sha256(args.g0a_results)}}
    write_metadata(out / "summary.json", summary)
    write_metadata(out / "decision.json", {"decision": summary["decision"], "training_authorized": False, "internal_used": False, "development_used": False, "confirmation_used": False})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
