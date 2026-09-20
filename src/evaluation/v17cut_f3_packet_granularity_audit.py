"""Read-only V8 packet-size versus fidelity-rollback audit."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.training.train_v17cut_b0_structured_fixed_v8 import prefix_masks
from src.training.train_v17h0_multianchor_cutoff import meets_fidelity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_TRAIN_ONLY_READ_ONLY":
        raise ValueError("unfrozen granularity audit")
    rows = [row for row in read_jsonl(args.candidates) if fold(row["example_id"]) != 4]
    if len(rows) != 1421:
        raise ValueError("unexpected train-only population")
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    transitions = []
    query_rows = []
    for number, row in enumerate(rows, 1):
        query = row["example_id"]
        order = orders[query]
        prefixes = prefix_masks(order)
        required = set(prefixes)
        exact = {}
        for record in read_jsonl(Path(args.exact_dir) / f"{query}.jsonl"):
            mask = state_to_mask(record["state"])
            if mask in required:
                exact[mask] = record
        if set(exact) != required:
            raise ValueError(f"missing V8 prefixes: {query}")
        fidelity = [float(exact[mask]["fidelity"]) for mask in prefixes]
        tokens = [int(exact[mask]["tokens"]) for mask in prefixes]
        increments = [tokens[depth] - tokens[depth - 1] for depth in range(1, 13)]
        if any(value <= 0 for value in increments):
            raise ValueError(f"nonpositive packet token increment: {query}")
        top_three = set(sorted(range(12), key=lambda position: (-increments[position], position))[:3])
        safe = [meets_fidelity(value, .9) for value in fidelity]
        first = next((depth for depth, hit in enumerate(safe) if hit), None)
        first_run_length = 0
        if first is not None:
            while first + first_run_length < 13 and safe[first + first_run_length]:
                first_run_length += 1
        terminator = first + first_run_length if first is not None and first + first_run_length < 13 else None
        for index in range(12):
            before, after = fidelity[index], fidelity[index + 1]
            transitions.append({
                "example_id": query, "depth_after": index + 1, "packet_id": order[index],
                "token_increment": increments[index], "large_within_query": index in top_three,
                "fidelity_before": before, "fidelity_after": after,
                "delta_fidelity": after - before,
                "rollback_090": safe[index] and not safe[index + 1],
                "large_negative_jump": after - before <= -.10,
                "first_window_terminator": terminator == index + 1,
            })
        query_rows.append({"example_id": query, "first_090_depth": first,
                           "first_success_run_length": first_run_length,
                           "first_window_terminator_depth": terminator})
        if number % 500 == 0:
            print(json.dumps({"loaded": number, "total": len(rows)}), flush=True)
    groups = {}
    for label, subset in (("top_three_large", [row for row in transitions if row["large_within_query"]]),
                          ("other_nine", [row for row in transitions if not row["large_within_query"]])):
        groups[label] = {"transitions": len(subset), "rollback_090": sum(row["rollback_090"] for row in subset),
                         "large_negative_jump": sum(row["large_negative_jump"] for row in subset),
                         "mean_token_increment": mean(row["token_increment"] for row in subset),
                         "median_token_increment": median(row["token_increment"] for row in subset)}
        groups[label]["rollback_090_rate"] = groups[label]["rollback_090"] / len(subset)
    rollback_depth = Counter(row["depth_after"] for row in transitions if row["rollback_090"])
    query_by_id = {row["example_id"]: row for row in query_rows}
    narrow = [row for row in transitions if row["first_window_terminator"] and
              query_by_id[row["example_id"]]["first_success_run_length"] == 1]
    # Query lookup is used only for grouping, never as a model feature.
    narrow_query_count = sum(row["first_success_run_length"] == 1 for row in query_rows)
    ratio = groups["top_three_large"]["rollback_090_rate"] / groups["other_nine"]["rollback_090_rate"] if groups["other_nine"]["rollback_090_rate"] else None
    passed = (sum(row["rollback_090"] for row in transitions) >= 20 and ratio is not None and ratio >= 2)
    result = {
        "protocol": config["protocol"], "queries": len(rows), "transitions": len(transitions),
        "size_groups": groups, "rollback_090_total": sum(row["rollback_090"] for row in transitions),
        "rollback_090_by_depth": dict(sorted(rollback_depth.items())),
        "large_to_other_rollback_rate_ratio": ratio,
        "queries_with_one_prefix_first_success_run": narrow_query_count,
        "one_prefix_terminators": len(narrow),
        "one_prefix_terminators_in_large_packets": sum(row["large_within_query"] for row in narrow),
        "bounded_split_pilot_design_gate_passed": passed,
        "causal_limit": "association between packet length and rollback cannot establish that splitting preserves or improves fidelity; half-packet states require new Target calls",
        "fold4_611_read": False, "holdout_581_read": False, "internal300_read": False,
        "development_read": False, "confirmation_read": False, "new_target_calls": 0,
        "artifacts": {"config_sha256": sha256(args.config), "candidates_sha256": sha256(args.candidates),
                      "rollouts_sha256": sha256(args.rollouts)},
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train1421_transitions.jsonl", transitions)
    write_jsonl(output / "train1421_queries.jsonl", query_rows)
    write_metadata(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
