from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata
from src.search.sequential_trajectory_dp import SequentialTrajectoryDP


def stable_seed(example_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{example_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def build_example(payload: tuple[dict, str, int, int, int]) -> dict:
    source, exact_dir, base_seed, oracle_rollouts, random_rollouts = payload
    example_id = source["example_id"]
    exact = list(
        read_jsonl(Path(exact_dir) / f"{example_id}.jsonl", ExactSearchResult)
    )
    dp = SequentialTrajectoryDP(exact, source["active_levels"])
    supervision = dp.one_run_supervision(
        seed=stable_seed(example_id, base_seed),
        oracle_rollouts=oracle_rollouts,
        random_rollouts=random_rollouts,
    )
    histories = [
        {
            "history": list(item.history),
            "selected_mask": item.selected_mask,
            "reached_levels": item.reached_levels,
            "optimal_actions": list(item.optimal_actions),
            "source": item.source,
        }
        for item in supervision
    ]
    output = {
        "example_id": example_id,
        "question": source["question"],
        "packet_ids": source["packet_ids"],
        "packet_texts": source["packet_texts"],
        "packet_tokens": source["packet_tokens"],
        "active_levels": source["active_levels"],
        "histories": histories,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed one-run V8 history supervision")
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--oracle-rollouts", type=int, default=4)
    parser.add_argument("--random-rollouts", type=int, default=4)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    forbidden = ("calibration300", "final_test300", "test300")
    if any(token in args.oracle.lower() for token in forbidden):
        raise ValueError("locked calibration/final-test roles must not enter V8 supervision")
    sources = list(read_jsonl(args.oracle))
    payloads = [
        (row, args.exact_dir, args.seed, args.oracle_rollouts, args.random_rollouts)
        for row in sources
    ]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(pool.map(build_example, payloads, chunksize=4), 1):
            rows.append(row)
            if index % 100 == 0 or index == len(sources):
                print(json.dumps({"processed": index, "total": len(sources)}), flush=True)
    if [row["example_id"] for row in rows] != [row["example_id"] for row in sources]:
        raise RuntimeError("parallel builder did not preserve source order")
    write_jsonl(args.output, rows)
    counts = [len(row["histories"]) for row in rows]
    source_counts: dict[str, int] = {}
    for row in rows:
        for history in row["histories"]:
            key = history["source"].split("_", 1)[0]
            source_counts[key] = source_counts.get(key, 0) + 1
    manifest = {
        "complete": True,
        "examples": len(rows),
        "histories": sum(counts),
        "minimum_histories_per_example": min(counts),
        "maximum_histories_per_example": max(counts),
        "mean_histories_per_example": sum(counts) / len(counts),
        "source_counts": source_counts,
        "seed": args.seed,
        "oracle_rollouts": args.oracle_rollouts,
        "single_deviation_rollouts": 12,
        "random_rollouts": args.random_rollouts,
        "action_slack": 0.0,
        "oracle": args.oracle,
        "oracle_sha256": sha256(args.oracle),
        "output": args.output,
        "output_sha256": sha256(args.output),
        "locked_roles_used": False,
    }
    write_metadata(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
