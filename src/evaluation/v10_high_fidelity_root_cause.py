"""Reproducible consumed-development audit of persistent fidelity-0.90 failures."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.data.schemas import read_jsonl
from src.reproducibility import sha256, write_metadata

DEFAULT_MODELS = {
    "v4": "results/v2_rank_then_cut/v4_selected_development_evaluation/details.jsonl",
    "v5a": "results/v2_rank_then_cut/v5a_selected_development_evaluation/details.jsonl",
    "v6": "results/v2_rank_then_cut/v6_selected_development_evaluation/details.jsonl",
    "v7": "results/v2_rank_then_cut/v7_selected_development_evaluation/details.jsonl",
    "v8_step250": "results/v2_rank_then_cut/v8_one_run_seed20260912/checkpoints/step250/development_details.jsonl",
    "v8_step500": "results/v2_rank_then_cut/v8_one_run_seed20260912/checkpoints/step500/development_details.jsonl",
    "v8_step750": "results/v2_rank_then_cut/v8_one_run_seed20260912/checkpoints/step750/development_details.jsonl",
    "v9a": "results/v2_rank_then_cut/v9a_cost_probe_step250/development_beam8_details.jsonl",
    "v9b": "results/v2_rank_then_cut/v9b_coverage_probe_step250/development_beam8_details.jsonl",
    "v10": "results/v2_rank_then_cut/v10_mask_value_probe_seed20260912/development_details.jsonl",
}


def success_at(row: dict[str, Any], level: float) -> bool:
    return bool(next(anchor["contract_success"] for anchor in row["anchors"] if float(anchor["fidelity_level"]) == level))


def random_path_hit_probability(positive: list[bool]) -> float:
    avoid = [0] * len(positive)
    avoid[0] = int(not positive[0])
    for mask, count in enumerate(avoid):
        if not count:
            continue
        for packet in range(12):
            if mask & (1 << packet):
                continue
            next_mask = mask | (1 << packet)
            if not positive[next_mask]:
                avoid[next_mask] += count
    return 1.0 - avoid[-1] / math.factorial(12)


def lattice_features(row: dict[str, Any]) -> dict[str, Any]:
    positive = [int(value["attained_levels"]) >= 4 for value in row["mask_values"]]
    masks = [mask for mask, value in enumerate(positive) if value]
    intersection = (1 << 12) - 1
    union = 0
    for mask in masks:
        intersection &= mask
        union |= mask
    required = sum(bool(intersection & (1 << bit)) for bit in range(12))
    forbidden = sum(not bool(union & (1 << bit)) for bit in range(12))
    up_edges = down_edges = 0
    for mask in range(4096):
        for packet in range(12):
            if mask & (1 << packet):
                continue
            following = positive[mask | (1 << packet)]
            up_edges += int(not positive[mask] and following)
            down_edges += int(positive[mask] and not following)
    return {
        "positive_masks": len(masks),
        "positive_fraction": len(masks) / 4096,
        "minimum_positive_mask_size": min(mask.bit_count() for mask in masks),
        "maximum_positive_mask_size": max(mask.bit_count() for mask in masks),
        "required_packets_in_every_positive_mask": required,
        "forbidden_packets_in_every_positive_mask": forbidden,
        "full_mask_positive": positive[-1],
        "upward_boundary_edges": up_edges,
        "harmful_downward_boundary_edges": down_edges,
        "random_order_hits_positive_prefix_probability": random_path_hit_probability(positive),
    }


def aggregate(ids: set[str], features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output = {"examples": len(ids), "groups": dict(Counter(value.split("__")[1] for value in ids))}
    for key in next(iter(features.values())):
        values = [features[example_id][key] for example_id in ids]
        output[key] = {"mean": mean(values), "median": median(values), "min": min(values), "max": max(values)}
    return output


def population(path: str) -> dict[str, Any]:
    counts = Counter()
    positive_counts = []
    full_negative = 0
    examples = 0
    for row in read_jsonl(path):
        examples += 1
        values = {int(key): int(value) for key, value in row["class_population"].items()}
        positive = values.get(4, 0) + values.get(5, 0)
        positive_counts.append(positive)
        counts["le_1" if positive <= 1 else "2_4" if positive <= 4 else "5_10" if positive <= 10 else "gt_10"] += 1
        full = next(item for item in row["mask_values"] if int(item["mask"]) == 4095)
        full_negative += int(int(full["attained_levels"]) < 4)
    return {
        "examples": examples,
        "positive_mask_count_mean": mean(positive_counts),
        "positive_mask_count_median": median(positive_counts),
        "positive_mask_count_strata": dict(counts),
        "full_mask_below_0.90": full_negative,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", default="results/v2_rank_then_cut/v10_development300_mask_value_full.jsonl")
    parser.add_argument("--train", default="results/v2_rank_then_cut/v10_train3163_mask_value.jsonl")
    parser.add_argument("--output", default="results/v2_rank_then_cut/v10_high_fidelity_root_cause.json")
    args = parser.parse_args()
    models = {
        label: {row["example_id"]: row for row in read_jsonl(path)}
        for label, path in DEFAULT_MODELS.items()
    }
    ids = set(next(iter(models.values())))
    if any(set(rows) != ids for rows in models.values()):
        raise ValueError("model evaluations do not cover identical examples")
    failure_sets = {
        label: {example_id for example_id, row in rows.items() if not success_at(row, 0.9)}
        for label, rows in models.items()
    }
    persistent = set.intersection(*failure_sets.values())
    ever_failed = set.union(*failure_sets.values())
    development = {row["example_id"]: row for row in read_jsonl(args.development)}
    if set(development) != ids:
        raise ValueError("development lattice IDs differ from model evaluations")
    features = {example_id: lattice_features(row) for example_id, row in development.items()}
    frequencies = Counter(example_id for failures in failure_sets.values() for example_id in failures)
    result = {
        "complete": True,
        "scientific_role": "post-stop consumed-development root-cause audit; not a selection or confirmation result",
        "models": {
            label: {"fidelity_0.90_successes": len(ids) - len(failures), "failures": len(failures)}
            for label, failures in failure_sets.items()
        },
        "persistent_failures_all_ten_models": sorted(persistent),
        "persistent_failure_count": len(persistent),
        "failure_frequency_histogram": dict(sorted(Counter(frequencies.values()).items())),
        "groups": {
            "persistent_all_ten": aggregate(persistent, features),
            "intermittent_failure": aggregate(ever_failed - persistent, features),
            "never_failed": aggregate(ids - ever_failed, features),
        },
        "population_coverage": {"train3163": population(args.train), "development300": population(args.development)},
        "v8_checkpoint_dynamics": {
            "step250_to_step500_rescued": len(failure_sets["v8_step250"] - failure_sets["v8_step500"]),
            "step250_to_step500_broken": len(failure_sets["v8_step500"] - failure_sets["v8_step250"]),
            "step500_to_step750_rescued": len(failure_sets["v8_step500"] - failure_sets["v8_step750"]),
            "step500_to_step750_broken": len(failure_sets["v8_step750"] - failure_sets["v8_step500"]),
        },
        "artifacts": {
            "development_sha256": sha256(args.development),
            "train_sha256": sha256(args.train),
            "model_detail_sha256": {label: sha256(path) for label, path in DEFAULT_MODELS.items()},
        },
    }
    write_metadata(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
