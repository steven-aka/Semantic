"""Train-only exact-cache audit of stable add-only trajectory headroom."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl, write_jsonl
from src.search.atomic_nested_chain import state_to_mask
from src.training.train_v17h0_multianchor_cutoff import LEVELS, meets_fidelity, project


def chain(order, fidelity, tokens, level):
    mask = 0
    masks = [mask]
    for packet in order:
        mask |= 1 << packet
        masks.append(mask)
    if len(masks) != 13 or masks[-1] != 4095 or len(set(masks)) != 13:
        raise ValueError("invalid add-only chain")
    safe = [meets_fidelity(fidelity[mask], level) for mask in masks]
    starts = [i for i, hit in enumerate(safe) if hit and (i == 0 or not safe[i - 1])]
    runs = []
    for start in starts:
        end = start
        while end + 1 < 13 and safe[end + 1]:
            end += 1
        runs.append((start, end))
    first = runs[0] if runs else None
    return {
        "success": bool(runs),
        "earliest_tokens": tokens[masks[first[0]]] if first else None,
        "first_window": first[1] - first[0] + 1 if first else 0,
        "max_window": max((end - start + 1 for start, end in runs), default=0),
        "persistent_from_first": bool(first and first[1] == 12),
        "rollbacks": sum(safe[i] and not safe[i + 1] for i in range(12)),
        "max_negative_jump": min((fidelity[masks[i + 1]] - fidelity[masks[i]] for i in range(12)), default=0),
        "first_window_token_span": tokens[masks[first[1]]] - tokens[masks[first[0]]] if first else None,
    }


def lattice(fidelity, tokens, level):
    """Exact subset-DAG safe-run DP; result is a per-anchor oracle ceiling."""
    run = bytearray(4096)
    for mask in range(4095, -1, -1):
        if not meets_fidelity(fidelity[mask], level):
            continue
        remaining = 4095 ^ mask
        best = 0
        while remaining:
            bit = remaining & -remaining
            best = max(best, run[mask | bit])
            remaining ^= bit
        run[mask] = best + 1
    return {
        "max_window": max(run),
        "min_start_tokens": [min((tokens[mask] for mask in range(4096) if run[mask] >= k), default=None) for k in (1, 2, 3)],
    }


def exact(path):
    fidelity, tokens = [None] * 4096, [None] * 4096
    for record in read_jsonl(path):
        mask = state_to_mask(record["state"])
        if fidelity[mask] is not None:
            raise ValueError(f"duplicate mask: {path} {mask}")
        fidelity[mask], tokens[mask] = float(record["fidelity"]), int(record["tokens"])
    if any(value is None for value in fidelity) or any(value is None for value in tokens):
        raise ValueError(f"incomplete lattice: {path}")
    return fidelity, tokens


def aggregate(rows, source, level):
    valid = [row[source][level] for row in rows if level in row[source]]
    reached = [row for row in valid if row["success"]]
    return {
        "defined": len(valid), "success": len(reached),
        "first_window_width_1": sum(row["first_window"] == 1 for row in reached),
        "first_window_width_2": sum(row["first_window"] == 2 for row in reached),
        "first_window_width_3plus": sum(row["first_window"] >= 3 for row in reached),
        "any_window_3plus": sum(row["max_window"] >= 3 for row in reached),
        "persistent_from_first": sum(row["persistent_from_first"] for row in reached),
        "rollback_events": sum(row["rollbacks"] for row in valid),
        "mean_earliest_token_fraction": mean(row[source][level]["earliest_tokens"] / row["full_tokens"] for row in rows if level in row[source] and row[source][level]["success"]) if reached else None,
        "median_earliest_token_fraction": median(row[source][level]["earliest_tokens"] / row["full_tokens"] for row in rows if level in row[source] and row[source][level]["success"]) if reached else None,
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "candidates", "rollouts", "exact-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_TRAIN_ONLY_READ_ONLY":
        raise ValueError("protocol not frozen")
    rows = [row for row in read_jsonl(args.candidates) if fold(row["example_id"]) != 4]
    if len(rows) != 1421:
        raise ValueError(f"unexpected query count: {len(rows)}")
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    details = []
    for number, row in enumerate(rows, 1):
        query = row["example_id"]
        base = orders[query]
        fidelity, tokens = exact(Path(args.exact_dir) / f"{query}.jsonl")
        if tokens[4095] != row["full_tokens"]:
            raise ValueError(f"token mismatch: {query}")
        attainable = sorted(set(float(x) for x in row["attainable_levels"]))
        if any(x not in LEVELS for x in attainable):
            raise ValueError(f"invalid anchors: {query}")
        candidates = [base] + [project(base, int(c["mask"])) for c in row["candidates"][1:11]]
        if len(candidates) != 11:
            raise ValueError(f"missing top10: {query}")
        item = {"example_id": query, "full_tokens": tokens[4095], "attainable_levels": attainable,
                "v8_complete": bool(row["candidates"][0]["complete"]),
                "pool_oracle_complete": any(bool(c["complete"]) for c in row["candidates"][:11]),
                "v8_oracle_complete_cumulative_tokens": row["candidates"][0]["oracle_ordered_complete_cumulative_tokens"],
                "pool_oracle_complete_cumulative_tokens": min((c["oracle_ordered_complete_cumulative_tokens"] for c in row["candidates"][:11] if c["complete"]), default=None),
                "v8": {}, "pool_oracle": {}, "lattice_oracle": {}}
        for level in attainable:
            key = str(level)
            results = [chain(order, fidelity, tokens, level) for order in candidates]
            item["v8"][key] = results[0]
            # Pool envelope: each statistic may come from a different candidate.
            successes = [result for result in results if result["success"]]
            item["pool_oracle"][key] = {
                "success": bool(successes),
                "earliest_tokens": min((result["earliest_tokens"] for result in successes), default=None),
                "first_window": max((result["first_window"] for result in successes), default=0),
                "max_window": max((result["max_window"] for result in successes), default=0),
                "persistent_from_first": any(result["persistent_from_first"] for result in successes),
                "rollbacks": min((result["rollbacks"] for result in results), default=0),
            }
            item["lattice_oracle"][key] = lattice(fidelity, tokens, level)
        details.append(item)
        if number % 100 == 0:
            print(json.dumps({"processed": number, "total": len(rows)}), flush=True)
    summary = {"protocol": config["protocol"], "queries": len(details), "roles": config["population"], "levels": {},
               "complete": {"v8_oracle": sum(row["v8_complete"] for row in details),
                            "pool_oracle": sum(row["pool_oracle_complete"] for row in details),
                            "v8_mean_cumulative_tokens_given_complete": mean(row["v8_oracle_complete_cumulative_tokens"] for row in details if row["v8_complete"]),
                            "pool_mean_cumulative_tokens_given_complete": mean(row["pool_oracle_complete_cumulative_tokens"] for row in details if row["pool_oracle_complete"]),
                            "paired_cumulative_saving_both_complete": mean(row["v8_oracle_complete_cumulative_tokens"] - row["pool_oracle_complete_cumulative_tokens"] for row in details if row["v8_complete"] and row["pool_oracle_complete"])},
               "limitations": ["Per-anchor full-lattice DP can select a different path for each anchor; it is not a joint Complete oracle.",
                               "Pool oracle can select a different order per query, and even a different order per reported metric; no learned deployment gain is implied.",
                               "Stability width is number of consecutive prefix states, not repeated Target-run reliability."]}
    for level in LEVELS:
        key = str(level)
        v8 = aggregate(details, "v8", key)
        pool = aggregate(details, "pool_oracle", key)
        lattice_rows = [row["lattice_oracle"][key] for row in details if key in row["lattice_oracle"]]
        summary["levels"][key] = {"v8": v8, "pool_oracle_envelope": pool,
                                   "lattice_per_anchor_ceiling": {"defined": len(lattice_rows),
                                       "any_window_1": sum(row["max_window"] >= 1 for row in lattice_rows),
                                       "any_window_2plus": sum(row["max_window"] >= 2 for row in lattice_rows),
                                       "any_window_3plus": sum(row["max_window"] >= 3 for row in lattice_rows),
                                       "mean_min_start_token_fraction_for_3run": mean(row["min_start_tokens"][2] / detail["full_tokens"] for detail in details if key in detail["lattice_oracle"] for row in [detail["lattice_oracle"][key]] if row["min_start_tokens"][2] is not None) if any(row["max_window"] >= 3 for row in lattice_rows) else None}}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "per_query.jsonl", details)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
