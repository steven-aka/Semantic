"""Pair old and current Target outputs for the frozen four-action train cache."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.v17exec_a1_fresh_depth10_cache import LEVELS, choose, masks_for, score


def main():
    parser = argparse.ArgumentParser()
    for name in ("fresh", "candidates", "rollouts", "rerun-pairs", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    fresh = list(read_jsonl(args.fresh))
    by_key = {(row["example_id"], row["mask"]): row for row in fresh}
    if len(fresh) != 5684 or len(by_key) != 5684:
        raise AssertionError("incomplete or duplicated fresh cache")
    query_ids = {item["example_id"] for item in fresh}
    source = {row["example_id"]: row for row in read_jsonl(args.candidates) if row["example_id"] in query_ids}
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts) if row["example_id"] in source}
    if len(source) != 1421 or len(orders) != 1421:
        raise AssertionError("wrong train population")
    f1_changed = sum(abs(row["f1"] - row["old_cache_f1"]) > 1e-9 for row in fresh)
    by_level = {}
    for level in LEVELS:
        eligible = {query for query, row in source.items() if level in {float(value) for value in row["attainable_levels"]}}
        rows = [row for row in fresh if row["example_id"] in eligible]
        old_to_new = Counter((row["old_cache_f1"] + 1e-6 >= level, row["f1"] + 1e-6 >= level) for row in rows)
        by_level[str(level)] = {"eligible_contexts": len(rows), "old_success_new_failure": old_to_new[(True, False)],
                                "old_failure_new_success": old_to_new[(False, True)],
                                "unchanged_success": old_to_new[(True, True)], "unchanged_failure": old_to_new[(False, False)]}
    opportunity = Counter()
    for query, row in source.items():
        active = {float(value) for value in row["attainable_levels"]}
        masks = masks_for(orders[query])
        old, new = [], []
        for mask in masks:
            record = by_key[(query, mask)]
            current = score(record, active)
            current["mask"] = mask
            previous = score({**record, "f1": record["old_cache_f1"]}, active)
            previous["mask"] = mask
            old.append(previous)
            new.append(current)
        for name, options in (("old", old), ("fresh", new)):
            best = choose(options)
            stay = options[0]
            opportunity[name + "_repair_090"] += best["hits"][3] is True and stay["hits"][3] is False
            opportunity[name + "_repair_complete"] += all(hit is not False for hit in best["hits"]) and any(hit is False for hit in stay["hits"])
            opportunity[name + "_any_repair"] += any(hit is True and old_hit is False for hit, old_hit in zip(best["hits"], stay["hits"]))
        opportunity["any_opportunity_changed"] += any(a["hits"] != b["hits"] for a, b in zip(old, new))
    paired = []
    for row in read_jsonl(args.rerun_pairs):
        query = row["example_id"]
        values = [by_key[(query, mask)]["f1"] for mask in row["masks"]]
        paired.append({"example_id": query, "fresh_f1": values, "prior_rerun_f1": row["reruns"][0],
                       "same_f1": all(abs(a - b) <= 1e-9 for a, b in zip(values, row["reruns"][0]))})
    result = {"protocol": "V17-EXEC-A1_FRESH_VS_HISTORICAL_FOUR_MASK_DRIFT",
              "contexts": len(fresh), "changed_f1_contexts": f1_changed,
              "by_level": by_level, "opportunity": dict(opportunity),
              "prior_12_pair_f1_agreement": sum(row["same_f1"] for row in paired),
              "prior_12_pair_disagreements": [row for row in paired if not row["same_f1"]],
              "limitations": ["Historical cache disagreement measures cross-run changes, not within-run aleatoric probability.",
                              "Old and fresh outcomes share training queries and a historically trained V8 order."]}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
