"""Independent train-side spot check of A3 endpoint labels and action identity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.data.schemas import read_jsonl
from src.training.train_v17traj_a3_1_boundary_editor_cv import canonical_choices, training_label

LEVELS = (0.6, 0.7, 0.8, 0.9, 0.95)
DEPTHS = (10, 10, 10, 10, 10)


def raw_cache(path: Path, wanted: set[int]):
    found = {}
    for row in read_jsonl(path):
        mask = sum(int(bit) << index for index, bit in enumerate(row["state"]))
        if mask in wanted:
            if mask in found:
                raise AssertionError("duplicate exact-cache mask")
            found[mask] = (float(row["fidelity"]), int(row["tokens"]))
    if set(found) != wanted:
        raise AssertionError("missing exact-cache mask")
    return found


def independent_outcome(order, cache, active):
    prefix = 0
    masks = [0]
    for packet in order:
        prefix |= 1 << packet
        masks.append(prefix)
    hits = tuple((cache[masks[depth]][0] + 1e-6 >= level) if level in active else None
                 for depth, level in zip(DEPTHS, LEVELS))
    cost = sum(cache[masks[depth]][1] for depth, level in zip(DEPTHS, LEVELS) if level in active)
    return hits, cost


def independent_label(outcome, stay):
    hits, cost = outcome
    old, old_cost = stay
    if any(a is False and b is True for a, b in zip(hits, old)):
        return -1
    improved = any(a is True and b is False for a, b in zip(hits, old))
    if cost <= old_cost and (improved or cost < old_cost):
        return 1
    if not improved and cost > old_cost:
        return -1
    return 0


def rendered(packet_texts, order):
    chosen = set(order[:10])
    return "\n\n".join(text.strip() for packet, text in enumerate(packet_texts) if packet in chosen)


def select_ids(oof, number):
    def ordered(items):
        return sorted(items, key=lambda row: hashlib.sha256(row["example_id"].encode()).hexdigest())
    edited = ordered(row for row in oof if row["choice"] != 0)
    stayed = ordered(row for row in oof if row["choice"] == 0)
    return {row["example_id"] for row in edited[: number // 2] + stayed[: number - number // 2]}


def audit(candidates, data, rollouts, oof, exact_dir, sample_size=64):
    chosen = select_ids(oof, sample_size)
    sources = {row["example_id"]: row for row in candidates if row["example_id"] in chosen}
    texts = {row["example_id"]: row["packet_texts"] for row in data if row["example_id"] in chosen}
    orders = {row["example_id"]: row["decoded_order"] for row in rollouts if row["example_id"] in chosen}
    if len(sources) != sample_size or len(texts) != sample_size or len(orders) != sample_size:
        raise AssertionError("sample joins incomplete")
    mismatch = []
    equivalent_distinct_masks = []
    label_histogram = {-1: 0, 0: 0, 1: 0}
    near_threshold = 0
    for query in sorted(chosen):
        source = sources[query]
        active = {float(level) for level in source["attainable_levels"]}
        actions = canonical_choices(orders[query])
        masks = [sum(1 << packet for packet in order[:10]) for order in actions]
        if len(set(masks)) != 4:
            raise AssertionError("endpoint masks not distinct")
        cache = raw_cache(exact_dir / f"{query}.jsonl", set(masks))
        outcomes = [independent_outcome(action, cache, active) for action in actions]
        observed = [training_label(item, outcomes[0]) for item in outcomes[1:]]
        independent = [independent_label(item, outcomes[0]) for item in outcomes[1:]]
        if observed != independent:
            mismatch.append({"example_id": query, "observed": observed, "independent": independent})
        for value in independent:
            label_histogram[value] += 1
        if any(abs(cache[mask][0] - level) <= 0.011 for mask in masks for level in active):
            near_threshold += 1
        rendered_inputs = [rendered(texts[query], action) for action in actions]
        if len(set(rendered_inputs)) < len(rendered_inputs):
            equivalent_distinct_masks.append(query)
    return {"protocol": "V17-ACT-C0_LEARNING_CHAIN_SPOT_AUDIT", "sample_queries": sample_size,
            "edited_queries": sample_size // 2, "near_threshold_queries": near_threshold,
            "actions": sample_size * 4, "non_stay_label_histogram": {str(k): v for k, v in label_histogram.items()},
            "label_mismatches": mismatch, "distinct_mask_same_rendered_context_queries": equivalent_distinct_masks,
            "limitations": ["Independent label arithmetic checks cache-to-label plumbing, not cache-to-Target reproducibility.",
                            "Only four fixed-depth endpoints are compared; other schedules may have different action equivalence."]}


def main():
    parser = argparse.ArgumentParser()
    for name in ("candidates", "data", "rollouts", "oof", "exact-dir", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = audit(read_jsonl(args.candidates), read_jsonl(args.data), read_jsonl(args.rollouts),
                   list(read_jsonl(args.oof)), Path(args.exact_dir))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
