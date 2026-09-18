from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata
from src.search.sequential_trajectory_dp import SequentialTrajectoryDP


def add_costs(source: dict[str, Any], exact_dir: str) -> tuple[dict[str, Any], dict[str, int]]:
    exact = list(
        read_jsonl(Path(exact_dir) / f"{source['example_id']}.jsonl", ExactSearchResult)
    )
    dp = SequentialTrajectoryDP(exact, source["active_levels"])
    histories = []
    critical_states = 0
    for history in source["histories"]:
        mask = int(history["selected_mask"])
        reached = int(history["reached_levels"])
        advantages = dp.action_advantages(mask, reached)
        by_packet = {item.packet: item for item in advantages}
        values: list[float | None] = []
        lost: list[int | None] = []
        for packet in range(dp.width):
            item = by_packet.get(packet)
            values.append(None if item is None else item.scalar)
            lost.append(None if item is None else item.lost_reachable_anchors)
        if any((value or 0) > 0 for value in lost):
            critical_states += 1
        expected_optimal = {
            packet for packet, value in enumerate(values) if value is not None and value <= 1e-12
        }
        if expected_optimal != set(history["optimal_actions"]):
            raise RuntimeError(f"cost labels disagree with optimal set for {source['example_id']}")
        histories.append(
            {
                **history,
                "action_advantages": values,
                "action_lost_reachable_anchors": lost,
            }
        )
    return {**source, "histories": histories}, {
        "histories": len(histories),
        "critical_states": critical_states,
    }


def _worker(payload: tuple[dict[str, Any], str]) -> tuple[dict[str, Any], dict[str, int]]:
    return add_costs(*payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach exact V9 action cost-to-go labels")
    parser.add_argument("--input", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    forbidden = ("calibration300", "final_test300", "test300")
    if any(token in args.input.lower() for token in forbidden):
        raise ValueError("locked calibration/final-test roles must not enter V9 supervision")
    sources = list(read_jsonl(args.input))
    output_rows = []
    totals = {"histories": 0, "critical_states": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        payloads = ((row, args.exact_dir) for row in sources)
        for index, (row, counts) in enumerate(pool.map(_worker, payloads, chunksize=4), 1):
            output_rows.append(row)
            for key in totals:
                totals[key] += counts[key]
            if index % 100 == 0 or index == len(sources):
                print(json.dumps({"processed": index, "total": len(sources)}), flush=True)
    write_jsonl(args.output, output_rows)
    manifest = {
        "complete": True,
        "examples": len(output_rows),
        **totals,
        "critical_state_fraction": totals["critical_states"] / totals["histories"],
        "cost": {
            "primary": "two units per lost reachable anchor",
            "secondary": "future cumulative-token difference divided by active anchors times maximum lattice tokens",
            "ordering": "primary dominates the complete secondary-cost range",
        },
        "input": args.input,
        "input_sha256": sha256(args.input),
        "output": args.output,
        "output_sha256": sha256(args.output),
        "locked_roles_used": False,
    }
    write_metadata(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
