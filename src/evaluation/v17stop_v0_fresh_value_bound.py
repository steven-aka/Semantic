"""Quality-preserving clairvoyant stopping bound on fresh V8 prefix chains."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl

LEVELS = (.60, .70, .80, .90, .95)


def main() -> None:
    cfg = json.loads(Path("configs/v17stop_v0_fresh_value_bound.json").read_text())
    if cfg["status"] != "FROZEN_READ_ONLY_TRAIN_SIDE":
        raise ValueError("unfrozen stopping bound")
    source = {r["example_id"]: r for r in read_jsonl(
        "results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/sel_c0_train_top10_candidates.jsonl")
        if fold(r["example_id"]) != 4}
    chain = defaultdict(dict)
    root = Path("results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
    for path in sorted(root.glob("shard_*_of_3/per_prefix.jsonl")):
        for row in read_jsonl(path):
            q, depth = row["example_id"], row["depth"]
            if depth in chain[q]:
                raise AssertionError("duplicate prefix")
            chain[q][depth] = row
    if len(source) != 1421 or len(chain) != 1421 or any(len(c) != 12 for c in chain.values()):
        raise AssertionError("incomplete train-side fresh chain")
    summary = {"protocol": cfg["protocol"], "queries": len(source), "schedules": {},
               "limitations": cfg["cost_accounting"] + " " + cfg["oracle_contract"]}
    details = []
    for name, schedule in cfg["baseline_schedules"].items():
        if len(schedule) != 5 or sorted(schedule) != schedule:
            raise AssertionError("invalid monotone schedule")
        records = []
        for q, row in source.items():
            active = {float(x) for x in row["attainable_levels"]}
            targets = [(level, schedule[i]) for i, level in enumerate(LEVELS) if level in active]
            base_cost = sum(chain[q][depth]["tokens"] for _, depth in targets)
            # Dynamic program over the last stop depth, preserving the exact
            # baseline success vector and the previous-level monotonicity rule.
            states = {0: (0, [])}
            base_success = []
            for level, baseline_depth in targets:
                success = chain[q][baseline_depth]["f1"] + 1e-6 >= level
                base_success.append(success)
                allowed = ([d for d in range(1, 13) if chain[q][d]["f1"] + 1e-6 >= level]
                           if success else [baseline_depth])
                next_states = {}
                for previous, (cost, path) in states.items():
                    for depth in allowed:
                        if depth < previous:
                            continue
                        candidate = (cost + chain[q][depth]["tokens"], path + [depth])
                        if depth not in next_states or candidate[0] < next_states[depth][0]:
                            next_states[depth] = candidate
                states = next_states
            if not states:
                raise AssertionError("baseline schedule must be feasible")
            best_cost, best_depths = min(states.values(), key=lambda x: (x[0], x[1]))
            if best_cost > base_cost:
                raise AssertionError("oracle worse than baseline")
            records.append({"example_id": q, "baseline_tokens": base_cost,
                            "clairvoyant_tokens": best_cost, "saving": base_cost - best_cost,
                            "eligible_levels": [level for level, _ in targets],
                            "baseline_success": base_success, "oracle_depths": best_depths})
        summary["schedules"][name] = {
            "depths": schedule,
            "mean_baseline_cumulative_context_tokens": mean(r["baseline_tokens"] for r in records),
            "mean_clairvoyant_cumulative_context_tokens": mean(r["clairvoyant_tokens"] for r in records),
            "mean_context_token_saving_upper_bound": mean(r["saving"] for r in records),
            "median_context_token_saving": median(r["saving"] for r in records),
            "queries_with_saving": sum(r["saving"] > 0 for r in records),
            "baseline_complete": sum(all(r["baseline_success"]) for r in records),
            "new_target_calls": 0,
        }
        details.extend({"schedule": name, **record} for record in records)
    out = Path("results/v2_rank_then_cut/v17stop_v0_fresh_value_bound")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "per_query.jsonl").open("w") as handle:
        for record in details:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps(summary["schedules"]), flush=True)


if __name__ == "__main__":
    main()
