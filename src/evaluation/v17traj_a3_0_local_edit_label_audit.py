"""Audit set-valued local-edit labels before training a residual editor."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.v17traj_a0_robust_chain_oracle_audit import exact
from src.evaluation.v17traj_a2_stop_aligned_local_oracle import local_orders, path
from src.training.train_v17h0_multianchor_cutoff import LEVELS, meets_fidelity


def fixed_outcome(masks, fidelity, tokens, depths, active):
    hits = tuple(meets_fidelity(fidelity[masks[depth]], level) if level in active else None
                 for depth, level in zip(depths, LEVELS))
    cumulative = sum(tokens[masks[depth]] for depth, level in zip(depths, LEVELS) if level in active)
    return hits, cumulative


def label(outcome, baseline):
    hits, cost = outcome
    base_hits, base_cost = baseline
    repair = any(hit is True and old is False for hit, old in zip(hits, base_hits))
    broken = any(hit is False and old is True for hit, old in zip(hits, base_hits))
    if not broken and cost <= base_cost and (repair or cost < base_cost):
        return "beneficial"
    if (broken and (repair or cost < base_cost)) or (repair and cost > base_cost):
        return "tradeoff"
    if broken or (not repair and cost > base_cost):
        return "harmful"
    return "neutral"


def bucket(count):
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 3:
        return "2-3"
    return "4+"


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "candidates", "rollouts", "exact-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_TRAIN_ONLY_READ_ONLY":
        raise ValueError("protocol not frozen")
    schedules = config["schedules"]
    if config["primary_schedule"] not in schedules:
        raise ValueError("missing primary")
    rows = [row for row in read_jsonl(args.candidates) if fold(row["example_id"]) != 4]
    if len(rows) != 1421:
        raise ValueError("unexpected train population")
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    details = []
    frequency = {name: Counter() for name in schedules}
    type_counts = {name: Counter() for name in schedules}
    for number, source in enumerate(rows, 1):
        query = source["example_id"]
        base = orders[query]
        fidelity, tokens = exact(Path(args.exact_dir) / f"{query}.jsonl")
        active = set(float(level) for level in source["attainable_levels"])
        if active not in (set(LEVELS[:4]), set(LEVELS)):
            raise ValueError("invalid attainable levels")
        candidates = local_orders(base)
        if len(candidates) != 77:
            raise ValueError("unexpected edit count")
        rank = {packet: index for index, packet in enumerate(base)}
        signatures = ["-".join(str(rank[packet]) for packet in order) for order in candidates]
        masks = [path(order) for order in candidates]
        item = {"example_id": query, "attainable_levels": sorted(active), "schedules": {}}
        for name, depths in schedules.items():
            outputs = [fixed_outcome(mask, fidelity, tokens, depths, active) for mask in masks]
            baseline = outputs[0]
            positives = {i for i in range(1, 77) if label(outputs[i], baseline) == "beneficial"}
            categories = Counter(label(result, baseline) for result in outputs[1:])
            unique_outcomes = {tuple(mask[depth] for depth in depths) for mask in masks}
            positive_outcomes = {tuple(masks[i][depth] for depth in depths) for i in positives}
            benefit_types = Counter()
            for i in positives:
                hits, cost = outputs[i]
                base_hits, base_cost = baseline
                repair = any(hit is True and old is False for hit, old in zip(hits, base_hits))
                benefit_types["repair_090"] += hits[3] is True and base_hits[3] is False
                benefit_types["repair_complete"] += all(hit is not False for hit in hits) and any(hit is False for hit in base_hits)
                benefit_types["token_only"] += not repair and cost < base_cost
                benefit_types["repair_and_saving"] += repair and cost < base_cost
                frequency[name][signatures[i]] += 1
            type_counts[name].update(benefit_types)
            item["schedules"][name] = {"beneficial_edit_indices": sorted(positives),
                "beneficial_signatures": sorted(signatures[i] for i in positives),
                "beneficial_count": len(positives), "beneficial_outcome_classes": len(positive_outcomes),
                "distinct_stop_mask_outcomes": len(unique_outcomes), "labels": dict(categories),
                "benefit_types": dict(benefit_types)}
        details.append(item)
        if number % 100 == 0:
            print(json.dumps({"processed": number, "total": len(rows)}), flush=True)
    summary = {"protocol": config["protocol"], "queries": len(details), "primary_schedule": config["primary_schedule"],
               "schedules": {}, "overlap": {},
               "limitations": ["A beneficial edit uses exact cached outcomes as a label, not a deployment input.",
                               "Many edits can share the same stop-mask outcome; raw edit count overstates distinct decisions.",
                               "Schedule overlap measures label consistency on design-exposed train queries, not learned generalization."]}
    for name in schedules:
        records = [row["schedules"][name] for row in details]
        total = sum(record["beneficial_count"] for record in records)
        entropy = -sum((count / total) * math.log(count / total) for count in frequency[name].values()) if total else None
        summary["schedules"][name] = {
            "queries_with_beneficial_edit": sum(record["beneficial_count"] > 0 for record in records),
            "beneficial_edit_count_total": total,
            "beneficial_count_buckets": dict(Counter(bucket(record["beneficial_count"]) for record in records)),
            "median_beneficial_count_when_positive": median(record["beneficial_count"] for record in records if record["beneficial_count"] > 0) if total else None,
            "mean_distinct_stop_mask_outcomes": mean(record["distinct_stop_mask_outcomes"] for record in records),
            "mean_beneficial_outcome_classes_when_positive": mean(record["beneficial_outcome_classes"] for record in records if record["beneficial_count"] > 0) if total else None,
            "label_counts": dict(sum((Counter(record["labels"]) for record in records), Counter())),
            "benefit_type_edit_counts": dict(type_counts[name]),
            "distinct_beneficial_structural_signatures": len(frequency[name]),
            "top_beneficial_signatures": frequency[name].most_common(10),
            "signature_entropy_nats": entropy,
            "signature_effective_count": math.exp(entropy) if entropy is not None else None}
    names = list(schedules)
    for left_index in range(len(names)):
        for right_index in range(left_index + 1, len(names)):
            left, right = names[left_index], names[right_index]
            pair = [(set(row["schedules"][left]["beneficial_signatures"]),
                     set(row["schedules"][right]["beneficial_signatures"])) for row in details]
            both = [(a, b) for a, b in pair if a and b]
            union = [(a, b) for a, b in pair if a or b]
            summary["overlap"][f"{left}__{right}"] = {
                "positive_both_queries": len(both), "positive_either_queries": len(union),
                "positive_query_overlap_fraction": len(both) / len(union) if union else None,
                "mean_jaccard_given_both": mean(len(a & b) / len(a | b) for a, b in both) if both else None,
                "queries_with_any_shared_beneficial_edit": sum(bool(a & b) for a, b in both)}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_query.jsonl", details)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
