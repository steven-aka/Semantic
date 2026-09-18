from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from src.data.schemas import ExactSearchResult, read_jsonl
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import best_binary_nested_chain
from src.search.rank_then_cut import best_prefix_nested_chain
from src.search.sequential_trajectory_dp import SequentialTrajectoryDP


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the exact history-aware action oracle before V8 training"
    )
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--normalized-slack", type=float, default=0.005)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.normalized_slack < 0:
        raise ValueError("normalized slack must be nonnegative")
    forbidden = ("calibration300", "final_test300", "test300")
    if any(token in args.oracle.lower() for token in forbidden):
        raise ValueError("locked calibration/final-test roles must not enter this preflight")

    sources = list(read_jsonl(args.oracle))
    if args.limit is not None:
        sources = sources[: args.limit]
    exact_matches = 0
    canonical_matches = 0
    history_ambiguous_masks = 0
    multi_history_masks = 0
    exact_action_sizes: list[int] = []
    slack_action_sizes: list[int] = []
    examples_with_history_dependence = 0
    by_dataset: Counter[str] = Counter()
    for index, source in enumerate(sources, 1):
        example_id = source["example_id"]
        exact_path = Path(args.exact_dir) / f"{example_id}.jsonl"
        exact = list(read_jsonl(exact_path, ExactSearchResult))
        levels = source["active_levels"]
        dp = SequentialTrajectoryDP(exact, levels)
        root_reached = dp.attained[0]
        root = dp.value(0, root_reached)
        root_cost = root_reached * dp.tokens[0] + root.additional_cumulative_tokens
        nested = best_binary_nested_chain(exact, levels)
        nested_cost = sum(int(row["tokens"]) for row in nested if row["feasible"])
        exact_matches += int(root.reached_levels == len(levels) and root_cost == nested_cost)

        rollout = dp.canonical_rollout()
        evaluated = best_prefix_nested_chain(exact, rollout["order"], levels)
        evaluated_cost = sum(int(row["tokens"]) for row in evaluated if row["feasible"])
        canonical_matches += int(
            sum(bool(row["feasible"]) for row in evaluated) == len(levels)
            and evaluated_cost == nested_cost
        )
        exact_action_sizes.extend(int(value) for value in rollout["action_set_sizes"])
        slack_rollout = dp.canonical_rollout(normalized_slack=args.normalized_slack)
        slack_action_sizes.extend(int(value) for value in slack_rollout["action_set_sizes"])

        reachable = dp.reachable_history_levels()
        example_depends_on_history = False
        for mask, histories in enumerate(reachable[:-1]):
            if len(histories) < 2:
                continue
            multi_history_masks += 1
            action_sets = {dp.optimal_actions(mask, reached) for reached in histories}
            if len(action_sets) > 1:
                history_ambiguous_masks += 1
                example_depends_on_history = True
        examples_with_history_dependence += int(example_depends_on_history)
        by_dataset[example_id.split("__")[1] if "__" in example_id else "unknown"] += 1
        if index % 25 == 0 or index == len(sources):
            print(json.dumps({"processed": index, "total": len(sources)}), flush=True)

    output = {
        "complete": True,
        "status": "consumed-development preflight; no model selection and no locked roles",
        "examples": len(sources),
        "datasets": dict(by_dataset),
        "objective": {
            "primary": "maximize number of fidelity anchors reached by any prefix",
            "secondary": "minimize cumulative tokens at first anchor crossings",
            "state": "(selected_packet_mask, highest_fidelity_anchor_reached_in_history)",
        },
        "oracle_equivalence": {
            "root_value_matches_existing_nested_chain": exact_matches,
            "canonical_rollout_matches_existing_nested_chain": canonical_matches,
        },
        "action_supervision": {
            "exact_mean_action_set_size": mean(exact_action_sizes),
            "exact_median_action_set_size": median(exact_action_sizes),
            "near_optimal_normalized_slack": args.normalized_slack,
            "near_optimal_mean_action_set_size": mean(slack_action_sizes),
            "near_optimal_median_action_set_size": median(slack_action_sizes),
        },
        "history_state_audit": {
            "masks_with_multiple_reachable_history_levels": multi_history_masks,
            "masks_whose_exact_optimal_action_set_depends_on_history": history_ambiguous_masks,
            "fraction_among_multi_history_masks": (
                history_ambiguous_masks / multi_history_masks if multi_history_masks else 0.0
            ),
            "examples_with_at_least_one_history_dependent_mask": examples_with_history_dependence,
        },
        "artifacts": {
            "oracle": args.oracle,
            "oracle_sha256": sha256(args.oracle),
            "exact_dir": args.exact_dir,
        },
    }
    write_metadata(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

