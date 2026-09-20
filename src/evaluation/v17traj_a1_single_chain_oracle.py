"""Exact joint stable coverage and nearest-V8 witness on train-only lattices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.v17traj_a0_robust_chain_oracle_audit import exact, lattice
from src.training.train_v17h0_multianchor_cutoff import LEVELS, meets_fidelity, project


def trajectory(order, fidelity, tokens, levels):
    masks = [0]
    for packet in order:
        masks.append(masks[-1] | (1 << packet))
    if len(masks) != 13 or masks[-1] != 4095 or len(set(masks)) != 13:
        raise ValueError("invalid order")
    result = {"order": order, "ordinary": [], "stable": [], "stable_start_tokens": [],
              "stable_confirm_tokens": [], "first_window": [], "rollback_events": [], "worst_rollback": []}
    for level in levels:
        values = [fidelity[mask] for mask in masks]
        safe = [meets_fidelity(value, level) for value in values]
        starts = [i for i in range(11) if all(safe[i:i + 3])]
        start = starts[0] if starts else None
        result["ordinary"].append(any(safe))
        result["stable"].append(start is not None)
        result["stable_start_tokens"].append(tokens[masks[start]] if start is not None else None)
        result["stable_confirm_tokens"].append(tokens[masks[start + 2]] if start is not None else None)
        first = next((i for i, ok in enumerate(safe) if ok), None)
        width = 0
        while first is not None and first + width < 13 and safe[first + width]:
            width += 1
        result["first_window"].append(width)
        jumps = [values[i + 1] - values[i] for i in range(12) if safe[i] and not safe[i + 1]]
        result["rollback_events"].append(len(jumps))
        result["worst_rollback"].append(min(jumps, default=0.0))
    result["joint_stable_complete"] = all(result["stable"])
    result["stable_cumulative_tokens"] = sum(result["stable_start_tokens"]) if result["joint_stable_complete"] else None
    result["stable_confirm_cumulative_tokens"] = sum(result["stable_confirm_tokens"]) if result["joint_stable_complete"] else None
    return result


def dominance(candidate, base):
    if not all(x >= y for x, y in zip(candidate["ordinary"], base["ordinary"])):
        return "ordinary_break"
    if not all(x >= y for x, y in zip(candidate["stable"], base["stable"])):
        return "stable_break"
    if any(x > y for x, y in zip(candidate["stable"], base["stable"])):
        return "stable_repair_no_break"
    if candidate["joint_stable_complete"] and base["joint_stable_complete"]:
        costs = list(zip(candidate["stable_start_tokens"], base["stable_start_tokens"]))
        if all(x <= y for x, y in costs) and any(x < y for x, y in costs):
            return "token_pareto_no_quality_change"
        if candidate["stable_cumulative_tokens"] < base["stable_cumulative_tokens"]:
            return "token_tradeoff_across_anchors"
    return "no_gain"


def nearest_stable_order(base, fidelity, highest):
    """Exact minimum inversion path reaching a three-prefix safe run."""
    inf = 10**9
    distance = [[inf] * 4096 for _ in range(4)]
    parent = [[None] * 4096 for _ in range(4)]
    rank = {packet: i for i, packet in enumerate(base)}
    earlier = {packet: sum(1 << q for q in base[:rank[packet]]) for packet in base}
    initial = 1 if meets_fidelity(fidelity[0], highest) else 0
    distance[initial][0] = 0
    for mask in range(4096):
        remaining = 4095 ^ mask
        for run in range(4):
            cost = distance[run][mask]
            if cost == inf:
                continue
            bits = remaining
            while bits:
                bit = bits & -bits
                packet = bit.bit_length() - 1
                next_mask = mask | bit
                next_run = 3 if run == 3 else (min(3, run + 1) if meets_fidelity(fidelity[next_mask], highest) else 0)
                next_cost = cost + (earlier[packet] & remaining).bit_count()
                if next_cost < distance[next_run][next_mask]:
                    distance[next_run][next_mask] = next_cost
                    parent[next_run][next_mask] = (run, packet)
                bits ^= bit
    if distance[3][4095] == inf:
        return None
    order = []
    mask, run = 4095, 3
    while mask:
        previous_run, packet = parent[run][mask]
        order.append(packet)
        mask ^= 1 << packet
        run = previous_run
    order.reverse()
    return order, distance[3][4095]


def inversion_distance(order, base):
    rank = {packet: index for index, packet in enumerate(base)}
    return sum(rank[order[i]] > rank[order[j]] for i in range(12) for j in range(i + 1, 12))


def first_divergence(order, base):
    return next((i for i, (x, y) in enumerate(zip(order, base), 1) if x != y), None)


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "candidates", "rollouts", "exact-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.config).read_text())
    if protocol["status"] != "FROZEN_TRAIN_ONLY_READ_ONLY":
        raise ValueError("protocol not frozen")
    candidates = [row for row in read_jsonl(args.candidates) if fold(row["example_id"]) != 4]
    if len(candidates) != 1421:
        raise ValueError("unexpected train-only population")
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    details = []
    for number, source in enumerate(candidates, 1):
        query = source["example_id"]
        base = orders[query]
        fidelity, tokens = exact(Path(args.exact_dir) / f"{query}.jsonl")
        levels = sorted(float(level) for level in source["attainable_levels"])
        if levels not in (list(LEVELS[:4]), list(LEVELS)):
            raise ValueError("invalid attainable levels")
        highest = levels[-1]
        raw_orders = [base] + [project(base, int(candidate["mask"])) for candidate in source["candidates"][1:11]]
        seen, pool = set(), []
        for order in raw_orders:
            key = tuple(order)
            if key not in seen:
                seen.add(key)
                pool.append(trajectory(order, fidelity, tokens, levels))
        v8 = pool[0]
        exact_reachable = lattice(fidelity, tokens, highest)["max_window"] >= 3
        if v8["joint_stable_complete"] and not exact_reachable:
            raise AssertionError("V8 exceeds lattice")
        if any(result["joint_stable_complete"] for result in pool) and not exact_reachable:
            raise AssertionError("pool exceeds lattice")
        # Nested scalar thresholds imply a stable run at the highest level
        # witnesses a valid single-chain stable run for every lower level.
        if v8["joint_stable_complete"] != v8["stable"][-1]:
            raise AssertionError("nested V8 levels violated")
        if any(result["joint_stable_complete"] != result["stable"][-1] for result in pool):
            raise AssertionError("nested pool levels violated")
        categories = [dominance(result, v8) for result in pool[1:]]
        witness = None
        if exact_reachable and not v8["joint_stable_complete"]:
            found = nearest_stable_order(base, fidelity, highest)
            if found is None:
                raise AssertionError("lattice witness missing")
            order, inversions = found
            tested = trajectory(order, fidelity, tokens, levels)
            if not tested["joint_stable_complete"] or inversion_distance(order, base) != inversions:
                raise AssertionError("invalid minimum-distance witness")
            witness = {"order": order, "inversions": inversions, "first_divergence_depth": first_divergence(order, base),
                       "trajectory": tested}
        details.append({"example_id": query, "defined_anchors": len(levels), "full_tokens": tokens[4095],
                        "v8": v8, "pool_size_unique": len(pool),
                        "pool_joint_stable_complete": any(result["joint_stable_complete"] for result in pool),
                        "pool_safe_stable_repair": "stable_repair_no_break" in categories,
                        "pool_token_pareto_no_quality_change": "token_pareto_no_quality_change" in categories,
                        "pool_best_complete_stable_cumulative_tokens": min((result["stable_cumulative_tokens"] for result in pool if result["joint_stable_complete"]), default=None),
                        "lattice_joint_stable_complete": exact_reachable, "nearest_lattice_witness": witness})
        if number % 100 == 0:
            print(json.dumps({"processed": number, "total": len(candidates)}), flush=True)
    summary = {"protocol": protocol["protocol"], "queries": len(details), "cohorts": {},
               "interpretation": ["Joint width-three coverage is exact because scalar fidelity thresholds are nested; no multi-anchor path switching is needed for coverage.",
                                  "A width-three future window is an oracle robustness surrogate, not a deployable stopping rule or required quality criterion.",
                                  "Minimum-Kendall witnesses certify the nearest single-chain structural repair, not minimum context tokens or cross-query learnability.",
                                  "Stable-run start is retrospectively known; confirmation requires observing two later prefixes and is substantially more expensive."]}
    for count in (4, 5):
        subset = [row for row in details if row["defined_anchors"] == count]
        witnesses = [row["nearest_lattice_witness"] for row in subset if row["nearest_lattice_witness"] is not None]
        both = [row for row in subset if row["v8"]["joint_stable_complete"] and row["pool_joint_stable_complete"]]
        summary["cohorts"][str(count)] = {"queries": len(subset),
            "v8_joint_stable_complete": sum(row["v8"]["joint_stable_complete"] for row in subset),
            "pool_joint_stable_complete": sum(row["pool_joint_stable_complete"] for row in subset),
            "lattice_joint_stable_complete": sum(row["lattice_joint_stable_complete"] for row in subset),
            "pool_safe_stable_repair": sum(row["pool_safe_stable_repair"] for row in subset),
            "pool_token_pareto_no_quality_change": sum(row["pool_token_pareto_no_quality_change"] for row in subset),
            "mean_paired_stable_cumulative_token_saving_both_complete": mean(row["v8"]["stable_cumulative_tokens"] - row["pool_best_complete_stable_cumulative_tokens"] for row in both) if both else None,
            "nearest_witness_queries": len(witnesses),
            "nearest_witness_inversion_median": median(w["inversions"] for w in witnesses) if witnesses else None,
            "nearest_witness_inversions_le_2": sum(w["inversions"] <= 2 for w in witnesses),
            "nearest_witness_first_divergence_median": median(w["first_divergence_depth"] for w in witnesses) if witnesses else None,
            "nearest_witness_first_divergence_le_3": sum(w["first_divergence_depth"] <= 3 for w in witnesses),
            "nearest_witness_new_ordinary_highest_success": sum(not row["v8"]["ordinary"][-1] for row in subset if row["nearest_lattice_witness"] is not None),
            "nearest_witness_mean_stable_start_fraction_highest": mean(row["nearest_lattice_witness"]["trajectory"]["stable_start_tokens"][-1] / row["full_tokens"] for row in subset if row["nearest_lattice_witness"] is not None) if witnesses else None,
            "nearest_witness_mean_stable_confirmation_fraction_highest": mean(row["nearest_lattice_witness"]["trajectory"]["stable_confirm_tokens"][-1] / row["full_tokens"] for row in subset if row["nearest_lattice_witness"] is not None) if witnesses else None}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_query.jsonl", details)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
