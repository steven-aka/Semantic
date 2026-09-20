"""Paired quality--context frontier for one-depth V8 stopping controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.training.train_v17cut_b0_structured_fixed_v8 import prefix_masks
from src.training.train_v17h0_multianchor_cutoff import LEVELS, meets_fidelity


def summarize(rows: list[dict], depth: int) -> dict:
    active = [[level in row["attainable_levels"] for level in LEVELS] for row in rows]
    hits = [[meets_fidelity(row["prefix_fidelity"][depth], level) if mask[index] else None
             for index, level in enumerate(LEVELS)] for row, mask in zip(rows, active)]
    complete = [all(hit is not False for hit in group) for group in hits]
    context = [row["prefix_tokens"][depth] / row["full_tokens"] for row in rows]
    regret = [
        (row["prefix_tokens"][depth] * sum(mask) - row["oracle_cost"]) / (sum(mask) * row["full_tokens"])
        for row, mask, good in zip(rows, active, complete) if good and row["oracle_cost"] is not None
    ]
    return {
        "depth": depth, "queries": len(rows),
        "anchor_eligible": [sum(mask[i] for mask in active) for i in range(5)],
        "anchor_success": [sum(group[i] is True for group in hits) for i in range(5)],
        "complete": sum(complete),
        "mean_final_context_fraction_all_queries": mean(context),
        "mean_final_context_fraction_complete_only": mean(value for value, good in zip(context, complete) if good) if any(complete) else None,
        "mean_normalized_regret_complete_only": mean(regret) if regret else None,
    }


def nondominated(points: list[dict]) -> list[int]:
    result = []
    for point in points:
        dominated = any(
            other is not point and other["anchor_success"][3] >= point["anchor_success"][3]
            and other["complete"] >= point["complete"]
            and other["mean_final_context_fraction_all_queries"] <= point["mean_final_context_fraction_all_queries"]
            and (other["anchor_success"][3] > point["anchor_success"][3]
                 or other["complete"] > point["complete"]
                 or other["mean_final_context_fraction_all_queries"] < point["mean_final_context_fraction_all_queries"])
            for other in points)
        if not dominated:
            result.append(point["depth"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_READ_ONLY":
        raise ValueError("unfrozen frontier protocol")
    source = list(read_jsonl(args.candidates))
    if len(source) != 2032:
        raise ValueError("unexpected candidate count")
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    rows = []
    for number, row in enumerate(source, 1):
        query = row["example_id"]
        prefixes = prefix_masks(orders[query])
        required = set(prefixes)
        values = {}
        for record in read_jsonl(Path(args.exact_dir) / f"{query}.jsonl"):
            mask = state_to_mask(record["state"])
            if mask in required:
                values[mask] = record
        if set(values) != required:
            raise ValueError(f"missing exact prefixes: {query}")
        rows.append({"example_id": query, "attainable_levels": row["attainable_levels"],
                     "full_tokens": row["full_tokens"],
                     "prefix_fidelity": [values[mask]["fidelity"] for mask in prefixes],
                     "prefix_tokens": [values[mask]["tokens"] for mask in prefixes],
                     "oracle_cost": row["candidates"][0]["oracle_ordered_complete_cumulative_tokens"]})
        if number % 500 == 0:
            print(json.dumps({"loaded": number, "total": len(source)}), flush=True)
    roles = {
        "train1421": [row for row in rows if fold(row["example_id"]) != 4],
        "design_exposed611": [row for row in rows if fold(row["example_id"]) == 4],
    }
    if len(roles["train1421"]) != 1421 or len(roles["design_exposed611"]) != 611:
        raise ValueError("unexpected query role split")
    summary = {}
    for role, records in roles.items():
        points = [summarize(records, depth) for depth in range(1, 13)]
        summary[role] = {"points": points, "nondominated_depths": nondominated(points)}
    result = {
        "protocol": config["protocol"], "roles": summary,
        "holdout_581_read": False, "internal300_read": False,
        "development_read": False, "confirmation_read": False, "new_target_calls": 0,
        "artifacts": {"config_sha256": sha256(args.config),
                      "candidates_sha256": sha256(args.candidates),
                      "rollouts_sha256": sha256(args.rollouts)},
    }
    write_metadata(args.output, result)
    print(json.dumps({role: {"nondominated_depths": value["nondominated_depths"],
                             "points": [{"depth": point["depth"], "success090": point["anchor_success"][3],
                                         "complete": point["complete"],
                                         "context": point["mean_final_context_fraction_all_queries"]}
                                        for point in value["points"]]}
                      for role, value in summary.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
