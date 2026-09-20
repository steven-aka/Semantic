"""Train-only exact-cache audit of depth-10 omitted packet pairs."""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.v17traj_a0_robust_chain_oracle_audit import exact
from src.evaluation.v17traj_a2_stop_aligned_local_oracle import local_orders, path
from src.training.train_v17h0_multianchor_cutoff import LEVELS, meets_fidelity


FULL = 4095
PAIRS = list(itertools.combinations(range(12), 2))


def scored(mask, fidelity, tokens, active):
    hits = [meets_fidelity(fidelity[mask], level) if level in active else None for level in LEVELS]
    return {"mask": mask, "hits": hits, "complete": all(hit is not False for hit in hits),
            "tokens": tokens[mask], "context_fraction": tokens[mask] / tokens[FULL]}


def choose(candidates, baseline):
    allowed = []
    for candidate in candidates:
        if candidate["tokens"] > baseline["tokens"]:
            continue
        if any(hit is False and old is True for hit, old in zip(candidate["hits"], baseline["hits"])):
            continue
        allowed.append(candidate)
    if not allowed:
        raise AssertionError("V8 baseline absent")
    return min(allowed, key=lambda item: (
        -sum(hit is True and old is False for hit, old in zip(item["hits"], baseline["hits"])),
        item["tokens"], item["mask"]))


def paired_summary(rows, name):
    chosen = [row[name] for row in rows]
    return {"anchor_success": [sum(item["hits"][i] is True for item in chosen) for i in range(5)],
            "complete": sum(item["complete"] for item in chosen),
            "mean_context_fraction": mean(item["context_fraction"] for item in chosen),
            "anchor_repairs": [sum(row[name]["hits"][i] is True and row["v8"]["hits"][i] is False for row in rows) for i in range(5)],
            "anchor_breaks": [sum(row[name]["hits"][i] is False and row["v8"]["hits"][i] is True for row in rows) for i in range(5)],
            "complete_repairs": sum(row[name]["complete"] and not row["v8"]["complete"] for row in rows),
            "complete_breaks": sum(not row[name]["complete"] and row["v8"]["complete"] for row in rows),
            "mean_paired_token_delta": mean(row[name]["tokens"] - row["v8"]["tokens"] for row in rows)}


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
        raise ValueError("unexpected population")
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    details = []
    interaction = {str(level): Counter() for level in LEVELS}
    interaction_values = {str(level): [] for level in LEVELS}
    best_pair_ranks = Counter()
    for number, source in enumerate(rows, 1):
        query = source["example_id"]
        fidelity, tokens = exact(Path(args.exact_dir) / f"{query}.jsonl")
        if tokens[FULL] != source["full_tokens"]:
            raise ValueError("full token mismatch")
        active = set(float(level) for level in source["attainable_levels"])
        if active not in (set(LEVELS[:4]), set(LEVELS)):
            raise ValueError("invalid attainable levels")
        base = orders[query]
        base_mask = path(base)[10]
        local_masks = {path(order)[10] for order in local_orders(base)}
        if len(local_masks) != 4:
            raise AssertionError("four local outcomes expected")
        all_masks = {FULL ^ (1 << i) ^ (1 << j) for i, j in PAIRS}
        if len(all_masks) != 66 or not local_masks <= all_masks or base_mask not in local_masks:
            raise AssertionError("invalid pair universe")
        base_score = scored(base_mask, fidelity, tokens, active)
        local_score = choose([scored(mask, fidelity, tokens, active) for mask in local_masks], base_score)
        global_score = choose([scored(mask, fidelity, tokens, active) for mask in all_masks], base_score)
        omitted = [packet for packet in base if not global_score["mask"] >> packet & 1]
        best_pair_ranks[tuple(sorted(base.index(packet) for packet in omitted))] += 1
        per_level = {}
        for level in active:
            key = str(level)
            if not meets_fidelity(fidelity[FULL], level):
                continue
            counts = Counter()
            values = []
            for i, j in PAIRS:
                first, second = FULL ^ (1 << i), FULL ^ (1 << j)
                pair = FULL ^ (1 << i) ^ (1 << j)
                safe_i = meets_fidelity(fidelity[first], level)
                safe_j = meets_fidelity(fidelity[second], level)
                safe_pair = meets_fidelity(fidelity[pair], level)
                counts["pairs"] += 1
                counts["both_single_safe_pair_unsafe"] += safe_i and safe_j and not safe_pair
                counts["both_single_unsafe_pair_safe"] += not safe_i and not safe_j and safe_pair
                counts["both_single_safe_pair_safe"] += safe_i and safe_j and safe_pair
                values.append(fidelity[pair] - fidelity[first] - fidelity[second] + fidelity[FULL])
            interaction[key].update(counts)
            interaction_values[key].extend(values)
            per_level[key] = {"both_single_safe_pair_unsafe": counts["both_single_safe_pair_unsafe"],
                              "both_single_unsafe_pair_safe": counts["both_single_unsafe_pair_safe"]}
        details.append({"example_id": query, "attainable_levels": sorted(active), "v8": base_score,
                        "local4_oracle": local_score, "all66_oracle": global_score,
                        "local4_available_repair": any(local_score["hits"][i] is True and base_score["hits"][i] is False for i in range(5)),
                        "all66_available_repair": any(global_score["hits"][i] is True and base_score["hits"][i] is False for i in range(5)),
                        "full12_interactions": per_level})
        if number % 100 == 0:
            print(json.dumps({"processed": number, "total": len(rows)}), flush=True)
    summary = {"protocol": config["protocol"], "queries": len(details),
               "v8": paired_summary(details, "v8"), "local4_oracle": paired_summary(details, "local4_oracle"),
               "all66_oracle": paired_summary(details, "all66_oracle"),
               "opportunity": {"queries_local4_any_anchor_repair": sum(row["local4_available_repair"] for row in details),
                               "queries_all66_any_anchor_repair": sum(row["all66_available_repair"] for row in details),
                               "queries_new_repair_only_outside_local4": sum(row["all66_available_repair"] and not row["local4_available_repair"] for row in details),
                               "most_common_oracle_omitted_v8_rank_pairs": [(list(pair), count) for pair, count in best_pair_ranks.most_common(10)]},
               "full12_deletion_interaction": {},
               "limitations": ["The full12 reference can fail despite a good depth10 subset; its single-deletion effects are not general packet-necessity labels.",
                               "An all66 depth10 subset oracle is not a jointly good progressive trajectory or a learnable policy.",
                               "All oracles choose with exact outcomes and cannot be deployed as reported."]}
    for level in LEVELS:
        key = str(level)
        values = interaction_values[key]
        summary["full12_deletion_interaction"][key] = {**dict(interaction[key]),
            "eligible_full12_success_queries": sum(key in row["full12_interactions"] for row in details),
            "mean_fidelity_difference_in_differences": mean(values) if values else None,
            "median_fidelity_difference_in_differences": median(values) if values else None}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "per_query.jsonl", details)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
