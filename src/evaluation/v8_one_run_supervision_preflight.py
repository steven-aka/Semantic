from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from src.data.schemas import ExactSearchResult, read_jsonl
from src.reproducibility import sha256, write_metadata
from src.search.sequential_trajectory_dp import SequentialTrajectoryDP


def stable_seed(example_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{example_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight the fixed expert/recovery history pool for one-run V8 training"
    )
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--oracle-rollouts", type=int, default=4)
    parser.add_argument("--random-rollouts", type=int, default=4)
    args = parser.parse_args()
    forbidden = ("calibration300", "final_test300", "test300")
    if any(token in args.oracle.lower() for token in forbidden):
        raise ValueError("locked calibration/final-test roles must not enter this preflight")

    sources = list(read_jsonl(args.oracle))
    states_per_example: list[int] = []
    unique_state_histories: list[int] = []
    action_sizes: list[int] = []
    recoverable = 0
    total = 0
    history_sources: Counter[str] = Counter()
    reached_counts: Counter[int] = Counter()
    for index, source in enumerate(sources, 1):
        example_id = source["example_id"]
        exact = list(
            read_jsonl(Path(args.exact_dir) / f"{example_id}.jsonl", ExactSearchResult)
        )
        dp = SequentialTrajectoryDP(exact, source["active_levels"])
        rows = dp.one_run_supervision(
            seed=stable_seed(example_id, args.seed),
            oracle_rollouts=args.oracle_rollouts,
            random_rollouts=args.random_rollouts,
        )
        states_per_example.append(len(rows))
        unique_state_histories.append(len({(row.selected_mask, row.reached_levels) for row in rows}))
        for row in rows:
            action_sizes.append(len(row.optimal_actions))
            history_sources[row.source.split("_", 1)[0]] += 1
            reached_counts[row.reached_levels] += 1
            recoverable += int(
                dp.value(row.selected_mask, row.reached_levels).reached_levels == len(dp.levels)
            )
            total += 1
        if index % 25 == 0 or index == len(sources):
            print(json.dumps({"processed": index, "total": len(sources)}), flush=True)

    output = {
        "complete": True,
        "status": "consumed-development offline-supervision preflight; no model training",
        "examples": len(sources),
        "protocol": {
            "seed": args.seed,
            "oracle_rollouts": args.oracle_rollouts,
            "single_nearest_nonoptimal_deviation_rollouts": 12,
            "random_rollouts": args.random_rollouts,
            "action_slack": 0.0,
            "deduplication_key": "ordered packet history",
        },
        "history_pool": {
            "total_unique_ordered_histories": total,
            "mean_per_example": mean(states_per_example),
            "median_per_example": median(states_per_example),
            "mean_unique_mask_history_states_per_example": mean(unique_state_histories),
            "mean_exact_action_set_size": mean(action_sizes),
            "median_exact_action_set_size": median(action_sizes),
            "fraction_from_which_all_active_levels_remain_reachable": recoverable / total,
            "first_seen_source_counts": dict(history_sources),
            "reached_level_counts": {str(key): value for key, value in sorted(reached_counts.items())},
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
