from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata
from src.search.sequential_trajectory_dp import SequentialTrajectoryDP


def augment_example(
    source: dict[str, Any], order: list[int], exact_dir: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sorted(order) != list(range(12)):
        raise ValueError("policy rollout must be a permutation of twelve packets")
    exact = list(
        read_jsonl(Path(exact_dir) / f"{source['example_id']}.jsonl", ExactSearchResult)
    )
    dp = SequentialTrajectoryDP(exact, source["active_levels"])
    fixed = {tuple(item["history"]) for item in source["histories"]}
    additions = []
    prefix_rows = []
    mask = 0
    reached = dp.attained[0]
    for depth, chosen in enumerate(order):
        history = tuple(order[:depth])
        advantages = dp.action_advantages(mask, reached)
        by_packet = {item.packet: item for item in advantages}
        values = [None if packet not in by_packet else by_packet[packet].scalar for packet in range(12)]
        lost = [
            None if packet not in by_packet else by_packet[packet].lost_reachable_anchors
            for packet in range(12)
        ]
        optimal = [packet for packet, value in enumerate(values) if value is not None and value <= 1e-12]
        additions.append(
            {
                "history": list(history),
                "selected_mask": mask,
                "reached_levels": reached,
                "optimal_actions": optimal,
                "source": "model_induced_v8_beam8",
                "action_advantages": values,
                "action_lost_reachable_anchors": lost,
            }
        )
        prefix_rows.append(
            {
                "depth": depth,
                "fixed_pool_covered": history in fixed,
                "chosen": chosen,
                "chosen_advantage": values[chosen],
                "chosen_lost_reachable_anchors": lost[chosen],
                "chosen_is_optimal": chosen in optimal,
            }
        )
        mask |= 1 << chosen
        reached = max(reached, dp.attained[mask])
    return {**source, "histories": [*source["histories"], *additions]}, {
        "example_id": source["example_id"],
        "prefixes": prefix_rows,
    }


def _worker(payload: tuple[dict[str, Any], list[int], str]):
    return augment_example(*payload)


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    prefixes = [prefix for row in details for prefix in row["prefixes"]]
    covered = [row for row in prefixes if row["fixed_pool_covered"]]
    uncovered = [row for row in prefixes if not row["fixed_pool_covered"]]

    def group(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "states": len(rows),
            "optimal_action_fraction": sum(row["chosen_is_optimal"] for row in rows) / len(rows),
            "lost_anchor_action_fraction": sum(
                int(row["chosen_lost_reachable_anchors"] or 0) > 0 for row in rows
            ) / len(rows),
            "mean_chosen_advantage": sum(float(row["chosen_advantage"] or 0.0) for row in rows)
            / len(rows),
        }

    return {
        "complete": True,
        "examples": len(details),
        "visited_states": len(prefixes),
        "fixed_pool_covered_states": len(covered),
        "fixed_pool_coverage_fraction": len(covered) / len(prefixes),
        "examples_with_uncovered_state": sum(
            any(not prefix["fixed_pool_covered"] for prefix in row["prefixes"]) for row in details
        ),
        "all": group(prefixes),
        "covered": group(covered),
        "uncovered": group(uncovered),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one round of exact V8 on-policy supervision")
    parser.add_argument("--input", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--details", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    forbidden = ("calibration300", "final_test300", "test300")
    if any(token in args.input.lower() for token in forbidden):
        raise ValueError("locked calibration/final-test roles must not enter supervision")
    sources = list(read_jsonl(args.input))
    rollout_rows = list(read_jsonl(args.rollouts))
    orders = {row["example_id"]: row["decoded_order"] for row in rollout_rows}
    if set(orders) != {row["example_id"] for row in sources}:
        raise ValueError("rollout IDs must exactly match supervision IDs")
    output_rows = []
    details = []
    payloads = ((row, orders[row["example_id"]], args.exact_dir) for row in sources)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (output, detail) in enumerate(pool.map(_worker, payloads, chunksize=4), 1):
            output_rows.append(output)
            details.append(detail)
            if index % 100 == 0 or index == len(sources):
                print(json.dumps({"processed": index, "total": len(sources)}), flush=True)
    summary = summarize(details)
    summary["artifacts"] = {
        "input": args.input,
        "input_sha256": sha256(args.input),
        "rollouts": args.rollouts,
        "rollouts_sha256": sha256(args.rollouts),
        "output": args.output,
        "locked_roles_used": False,
    }
    write_jsonl(args.output, output_rows)
    write_jsonl(args.details, details)
    summary["artifacts"]["output_sha256"] = sha256(args.output)
    summary["artifacts"]["details_sha256"] = sha256(args.details)
    write_metadata(args.summary, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
