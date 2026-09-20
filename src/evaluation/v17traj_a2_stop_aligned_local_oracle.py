"""Fixed-stop, train-only Pareto audit for Kendall<=2 V8 packet edits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.v17traj_a0_robust_chain_oracle_audit import exact
from src.training.train_v17h0_multianchor_cutoff import LEVELS, meets_fidelity


def local_orders(base):
    """All distinct orders at Kendall distance at most two from base."""
    if sorted(base) != list(range(12)):
        raise ValueError("invalid V8 order")
    found = {tuple(base)}
    one = set()
    for index in range(11):
        trial = list(base)
        trial[index], trial[index + 1] = trial[index + 1], trial[index]
        one.add(tuple(trial))
    found.update(one)
    for order in one:
        for index in range(11):
            trial = list(order)
            trial[index], trial[index + 1] = trial[index + 1], trial[index]
            found.add(tuple(trial))
    return [tuple(base)] + sorted(found - {tuple(base)})


def path(order):
    masks = [0]
    for packet in order:
        masks.append(masks[-1] | (1 << packet))
    if len(masks) != 13 or masks[-1] != 4095 or len(set(masks)) != 13:
        raise ValueError("invalid path")
    return masks


def score(order, masks, fidelity, tokens, depths, active):
    hits = [meets_fidelity(fidelity[masks[depth]], level) if level in active else None
            for depth, level in zip(depths, LEVELS)]
    costs = [tokens[masks[depth]] if level in active else None for depth, level in zip(depths, LEVELS)]
    width3 = [any(all(meets_fidelity(fidelity[masks[i + offset]], level) for offset in range(3))
                  for i in range(11)) if level in active else None for level in LEVELS]
    return {"order": list(order), "hits": hits, "costs": costs,
            "width3": width3,
            "complete": all(hit is not False for hit in hits),
            "cumulative_tokens": sum(value for value in costs if value is not None),
            "mean_context_fraction": mean(value / tokens[4095] for value in costs if value is not None)}


def safe(candidate, baseline, cost_bound):
    for hit, base_hit in zip(candidate["hits"], baseline["hits"]):
        if base_hit is True and hit is not True:
            return False
    return not cost_bound or candidate["cumulative_tokens"] <= baseline["cumulative_tokens"]


def choose(scores, baseline, cost_bound):
    allowed = [candidate for candidate in scores if safe(candidate, baseline, cost_bound)]
    if not allowed:
        raise AssertionError("STAY missing")
    return min(allowed, key=lambda candidate: (
        -sum(hit is True and base_hit is False for hit, base_hit in zip(candidate["hits"], baseline["hits"])),
        candidate["cumulative_tokens"], candidate["order"] != baseline["order"], candidate["order"]))


def summarize(rows, source):
    selected = [row[source] for row in rows]
    baseline = [row["v8"] for row in rows]
    eligible = [sum(row["v8"]["hits"][index] is not None for row in rows) for index in range(5)]
    successes = [sum(result["hits"][index] is True for result in selected) for index in range(5)]
    stable = [sum(result["width3"][index] is True for result in selected) for index in range(5)]
    repairs = [sum(row[source]["hits"][index] is True and row["v8"]["hits"][index] is False for row in rows) for index in range(5)]
    breaks = [sum(row[source]["hits"][index] is False and row["v8"]["hits"][index] is True for row in rows) for index in range(5)]
    both = [row for row in rows if row[source]["complete"] and row["v8"]["complete"]]
    return {"eligible": eligible, "anchor_success": successes, "width3_diagnostic": stable,
            "mean_context_fraction_by_anchor": [mean(result["costs"][index] / row["full_tokens"]
                for result, row in zip(selected, rows) if result["costs"][index] is not None) if eligible[index] else None
                for index in range(5)],
            "complete": sum(result["complete"] for result in selected),
            "complete_repairs": sum(row[source]["complete"] and not row["v8"]["complete"] for row in rows),
            "complete_breaks": sum(not row[source]["complete"] and row["v8"]["complete"] for row in rows),
            "anchor_repairs": repairs, "anchor_breaks": breaks,
            "edited_queries": sum(row[source]["order"] != row["v8"]["order"] for row in rows),
            "mean_normalized_cumulative_context": mean(result["mean_context_fraction"] for result in selected),
            "mean_cumulative_tokens": mean(result["cumulative_tokens"] for result in selected),
            "mean_paired_cumulative_token_delta_all": mean(row[source]["cumulative_tokens"] - row["v8"]["cumulative_tokens"] for row in rows),
            "mean_paired_cumulative_token_delta_both_complete": mean(row[source]["cumulative_tokens"] - row["v8"]["cumulative_tokens"] for row in both) if both else None}


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "candidates", "rollouts", "exact-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_TRAIN_ONLY_READ_ONLY":
        raise ValueError("unfrozen protocol")
    schedules = config["schedules"]
    for depths in schedules.values():
        if len(depths) != 5 or any(depth < 1 or depth > 12 for depth in depths) or depths != sorted(depths):
            raise ValueError("invalid fixed stopping schedule")
    sources = [row for row in read_jsonl(args.candidates) if fold(row["example_id"]) != 4]
    if len(sources) != 1421:
        raise ValueError("unexpected train-only population")
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    all_results = {name: [] for name in schedules}
    for number, source in enumerate(sources, 1):
        query = source["example_id"]
        fidelity, tokens = exact(Path(args.exact_dir) / f"{query}.jsonl")
        if tokens[4095] != source["full_tokens"]:
            raise ValueError("full-context token mismatch")
        active = set(float(level) for level in source["attainable_levels"])
        if active not in (set(LEVELS[:4]), set(LEVELS)):
            raise ValueError("invalid attainable levels")
        orders_and_masks = [(order, path(order)) for order in local_orders(orders[query])]
        for name, depths in schedules.items():
            candidates = [score(order, masks, fidelity, tokens, depths, active) for order, masks in orders_and_masks]
            base = candidates[0]
            constrained = choose(candidates, base, True)
            quality_first = choose(candidates, base, False)
            all_results[name].append({"example_id": query, "active_levels": sorted(active),
                                       "full_tokens": tokens[4095], "edits_examined": len(candidates), "v8": base,
                                       "cost_bounded_oracle": constrained,
                                       "quality_first_oracle": quality_first})
        if number % 100 == 0:
            print(json.dumps({"processed": number, "total": len(sources)}), flush=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"protocol": config["protocol"], "queries": len(sources), "schedules": {},
               "limitations": ["The stopping schedule is fixed before each query, but oracle local-edit selection still uses exact outcome labels and is not deployable.",
                               "At fixed depth an edit can change the selected packet subset, so context tokens are measured from the exact cache for each order.",
                               "These train-side opportunities do not establish held-out cross-query learnability or a final Pareto improvement."]}
    for name, rows in all_results.items():
        write_jsonl(output / f"{name}_per_query.jsonl", rows)
        summary["schedules"][name] = {"depths": schedules[name], "v8": summarize(rows, "v8"),
                                      "cost_bounded_oracle": summarize(rows, "cost_bounded_oracle"),
                                      "quality_first_oracle": summarize(rows, "quality_first_oracle")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
