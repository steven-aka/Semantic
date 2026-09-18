from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from src.data.schemas import ExactSearchResult, read_jsonl
from src.evaluation.v16c0_first_irreversible_divergence import trajectory_levels
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import best_binary_nested_chain
from src.search.rank_then_cut import best_prefix_nested_chain


def read_any_jsonl(path: str) -> list[dict[str, Any]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_orders(data: Sequence[dict[str, Any]], orders: dict[str, Sequence[int]], exact_dir: str) -> dict[str, Any]:
    per_level: dict[float, list[bool]] = {}; complete = 0; regrets = []
    for source in data:
        levels = trajectory_levels(source)
        exact = list(read_jsonl(Path(exact_dir) / f"{source['example_id']}.jsonl", ExactSearchResult))
        oracle = best_binary_nested_chain(exact, levels)
        learned = best_prefix_nested_chain(exact, orders[source["example_id"]], levels)
        successes = [bool(row["feasible"]) for row in learned]
        for level, success in zip(levels, successes): per_level.setdefault(level, []).append(success)
        if all(successes):
            complete += 1
            full_tokens = next(row.tokens for row in exact if all(row.state))
            learned_tokens = sum(int(row["tokens"]) for row in learned)
            oracle_tokens = sum(int(row["tokens"]) for row in oracle)
            regrets.append((learned_tokens - oracle_tokens) / (len(levels) * full_tokens))
    return {
        "examples": len(data), "complete_trajectory_successes": complete,
        "mean_complete_regret": mean(regrets) if regrets else None,
        "per_level": {str(level): {"examples": len(values), "successes": sum(values)} for level, values in sorted(per_level.items())},
        "semantics": "attainable_levels; absent anchors undefined, never negative",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Recompute a frozen decoded-order baseline with corrected attainable-level semantics")
    p.add_argument("--data", required=True); p.add_argument("--orders", required=True); p.add_argument("--exact-dir", required=True); p.add_argument("--output", required=True); p.add_argument("--label", required=True)
    args = p.parse_args(); data = list(read_jsonl(args.data)); detail = read_any_jsonl(args.orders)
    orders = {row["example_id"]: row["decoded_order"] for row in detail}
    result = summarize_orders(data, orders, args.exact_dir)
    result.update({"complete": True, "label": args.label, "data": args.data, "orders": args.orders, "data_sha256": sha256(args.data), "orders_sha256": sha256(args.orders), "locked_roles_used": False})
    write_metadata(args.output, result); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
