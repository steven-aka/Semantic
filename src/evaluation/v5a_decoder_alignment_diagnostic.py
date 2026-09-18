from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.evaluation.rank_projection_diagnostic import (
    _evaluate_order,
    _summarize_arm,
    stable_partial_order_projection,
)
from src.evaluation.v4_tail_cost_diagnostic import _closest_repair
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import best_binary_nested_chain


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose V5-A raw-edge versus global-decoder alignment"
    )
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--details", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", type=float, default=0.9)
    args = parser.parse_args()

    sources = {row["example_id"]: row for row in read_jsonl(args.oracle)}
    predictions = {row["example_id"]: row for row in read_jsonl(args.details)}
    if set(sources) != set(predictions):
        raise ValueError("oracle and prediction IDs differ")

    contingency: Counter[str] = Counter()
    rate_contingency: Counter[str] = Counter()
    raw_rows = []
    projected_rows = []
    repair_rows = []
    target_rescues = target_breaks = 0
    trajectory_rescues = trajectory_breaks = 0
    for example_id, source in sources.items():
        prediction = predictions[example_id]
        logits = prediction["precedence_logits"]
        order = tuple(int(value) for value in prediction["decoded_order"])
        position = {packet: index for index, packet in enumerate(order)}
        safety = [tuple(int(value) for value in pair) for pair in source["pairwise_preferences"]]
        for winner, loser in safety:
            raw_correct = float(logits[winner][loser]) > 0
            decoded_correct = position[winner] < position[loser]
            contingency[f"raw_{int(raw_correct)}_decoded_{int(decoded_correct)}"] += 1
        for edge in source["rate_preferences"]:
            winner, loser = int(edge["winner"]), int(edge["loser"])
            raw_correct = float(logits[winner][loser]) > 0
            decoded_correct = position[winner] < position[loser]
            rate_contingency[f"raw_{int(raw_correct)}_decoded_{int(decoded_correct)}"] += 1

        exact = list(
            read_jsonl(Path(args.exact_dir) / f"{example_id}.jsonl", ExactSearchResult)
        )
        levels = [float(value) for value in source["active_levels"]]
        oracle = best_binary_nested_chain(exact, levels)
        oracle_tokens = [int(row["tokens"]) for row in oracle]
        raw = _evaluate_order(exact, order, levels, oracle_tokens)
        projected_order = stable_partial_order_projection(order, safety)
        projected = _evaluate_order(exact, projected_order, levels, oracle_tokens)
        raw_rows.append({"example_id": example_id, **raw})
        projected_rows.append(
            {"example_id": example_id, "projected_order": list(projected_order), **projected}
        )
        raw_target = next(
            anchor["contract_success"]
            for anchor in raw["anchors"]
            if float(anchor["fidelity_level"]) == args.target
        )
        projected_target = next(
            anchor["contract_success"]
            for anchor in projected["anchors"]
            if float(anchor["fidelity_level"]) == args.target
        )
        target_rescues += int(not raw_target and projected_target)
        target_breaks += int(raw_target and not projected_target)
        trajectory_rescues += int(
            not raw["all_active_contracts_success"]
            and projected["all_active_contracts_success"]
        )
        trajectory_breaks += int(
            raw["all_active_contracts_success"]
            and not projected["all_active_contracts_success"]
        )

        if not raw_target:
            repair = _closest_repair(source, exact, order, args.target)
            relation_rows = []
            safety_set = set(safety)
            for winner in repair["missing_packets"]:
                for loser in repair["intruder_packets"]:
                    raw_supports = float(logits[winner][loser]) > 0
                    relation_rows.append(
                        {
                            "winner": winner,
                            "loser": loser,
                            "raw_logit": float(logits[winner][loser]),
                            "raw_supports_repair": raw_supports,
                            "safety_supervision": (
                                "supports_repair"
                                if (winner, loser) in safety_set
                                else "opposes_repair"
                                if (loser, winner) in safety_set
                                else "unlabeled"
                            ),
                        }
                    )
            repair_rows.append({"example_id": example_id, **repair, "relations": relation_rows})

    repair_relations = [edge for row in repair_rows for edge in row["relations"]]
    raw_correct_decoded_wrong = contingency["raw_1_decoded_0"]
    summary = {
        "complete": True,
        "status": "consumed-development oracle-label diagnostic only",
        "artifacts": {
            "oracle": args.oracle,
            "oracle_sha256": sha256(args.oracle),
            "details": args.details,
            "details_sha256": sha256(args.details),
        },
        "safety_edge_alignment": {
            "contingency": dict(contingency),
            "edges": sum(contingency.values()),
            "raw_sign_accuracy": (
                contingency["raw_1_decoded_0"] + contingency["raw_1_decoded_1"]
            )
            / sum(contingency.values()),
            "decoded_order_accuracy": (
                contingency["raw_0_decoded_1"] + contingency["raw_1_decoded_1"]
            )
            / sum(contingency.values()),
            "raw_correct_but_decoder_sacrificed": raw_correct_decoded_wrong,
            "fraction_of_raw_correct_edges_sacrificed": raw_correct_decoded_wrong
            / (contingency["raw_1_decoded_0"] + contingency["raw_1_decoded_1"]),
        },
        "rate_edge_alignment": {
            "contingency": dict(rate_contingency),
            "edges": sum(rate_contingency.values()),
        },
        "raw": _summarize_arm(raw_rows),
        "oracle_safety_projection": _summarize_arm(projected_rows),
        "projection_transitions": {
            "target_anchor_rescues": target_rescues,
            "target_anchor_breaks": target_breaks,
            "trajectory_rescues": trajectory_rescues,
            "trajectory_breaks": trajectory_breaks,
        },
        "failed_target_repairs": {
            "examples": len(repair_rows),
            "relations": len(repair_relations),
            "raw_supports_repair": sum(edge["raw_supports_repair"] for edge in repair_relations),
            "raw_opposes_repair": sum(not edge["raw_supports_repair"] for edge in repair_relations),
            "supervision": dict(Counter(edge["safety_supervision"] for edge in repair_relations)),
            "mean_absolute_logit": mean(abs(edge["raw_logit"]) for edge in repair_relations),
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "repair_edges.jsonl", repair_rows)
    write_jsonl(output / "oracle_safety_projection_details.jsonl", projected_rows)
    write_metadata(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
