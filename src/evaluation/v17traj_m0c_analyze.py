"""Post-run uncertainty and historical fixed-schedule comparison for M0C."""
import glob
import itertools
import json
import random
from pathlib import Path

import numpy as np

from src.data.schemas import read_jsonl

BASE = Path("results/v2_rank_then_cut")
OUT = BASE / "v17traj_m0c_borrow_pilot"
LEVELS = (0.60, 0.70, 0.80, 0.90, 0.95)


def main():
    rows = list(read_jsonl(OUT / "per_query.jsonl"))
    assert len(rows) == 128
    repairs = [sum(r["baseline_hits"][i] is False and r["action_hits"][i] is True for r in rows)
               for i in range(5)]
    breaks = [sum(r["baseline_hits"][i] is True and r["action_hits"][i] is False for r in rows)
              for i in range(5)]
    rng = random.Random(20260921)
    net = []
    for _ in range(10000):
        draw = [rows[rng.randrange(len(rows))] for _ in rows]
        net.append(sum(r["action_complete"] - r["baseline_complete"] for r in draw))
    net.sort()
    qids = [r["example_id"] for r in rows]
    position = {q: i for i, q in enumerate(qids)}
    source = {r["example_id"]: {float(x) for x in r["attainable_levels"]}
              for r in read_jsonl(BASE / "v17sel_b2b_lineage_clean_holdout/data/v10_v12_train_train_clean.jsonl")
              if r["example_id"] in position}
    f1 = np.empty((len(rows), 12))
    context = np.empty((len(rows), 12))
    for path in glob.glob(str(BASE / "v17canon_p0_fresh_v8_prefix_chain/shard_*_of_3/per_prefix.jsonl")):
        for item in read_jsonl(path):
            if item["example_id"] in position:
                i, d = position[item["example_id"]], item["depth"] - 1
                f1[i, d], context[i, d] = item["f1"], item["tokens"]
    attainable = np.array([[level in source[q] for level in LEVELS] for q in qids])
    action_success = np.array([sum(r["action_hits"][i] is True for r in rows) for i in range(5)])
    action_complete = sum(r["action_complete"] for r in rows)
    action_cost = sum(r["action_cost"]["context"] for r in rows) / len(rows)
    descriptive_caps = {}
    for cap in (16, 24, 32, 48):
        selected = [r for r in rows if r["depth7_slack"] <= cap]
        safe = [r for r in selected if not r["baseline_complete"] and r["action_complete"]
                and r["anchor_breaks"] == 0]
        descriptive_caps[str(cap)] = {
            "uniform_queries": len(selected),
            "uniform_complete_repairs": sum(not r["baseline_complete"] and r["action_complete"] for r in selected),
            "uniform_complete_breaks": sum(r["baseline_complete"] and not r["action_complete"] for r in selected),
            "uniform_mean_extra_cumulative_context": sum(r["cumulative_slack"] for r in selected) / len(rows),
            "hindsight_safe_complete_repairs": len(safe),
            "hindsight_safe_mean_extra_cumulative_context": sum(r["cumulative_slack"] for r in safe) / len(rows),
        }
    dominating = []
    references = {}
    for schedule in itertools.combinations_with_replacement(range(12), 5):
        success = np.stack([(f1[:, depth] + 1e-6 >= level) & attainable[:, i]
                            for i, (depth, level) in enumerate(zip(schedule, LEVELS))], axis=1)
        counts = success.sum(axis=0)
        complete = np.logical_or(~attainable, success).all(axis=1).sum()
        cost = (context[:, list(schedule)] * attainable).sum() / len(rows)
        if cost <= action_cost + 1e-6 and np.all(counts >= action_success) and complete >= action_complete:
            dominating.append([d + 1 for d in schedule])
        depths = tuple(d + 1 for d in schedule)
        if depths in ((6, 7, 7, 9, 10), (6, 8, 8, 9, 10), (7, 8, 8, 9, 10),
                      (6, 7, 7, 10, 10)):
            references[str(depths)] = {"five_anchor_success": counts.tolist(),
                                       "complete": int(complete), "mean_cumulative_context": float(cost)}
    result = {"protocol": "TRAJ-M0C_POST_RUN_DIAGNOSTICS",
              "per_anchor_repairs": repairs, "per_anchor_breaks": breaks,
              "paired_query_bootstrap_95pct_net_complete_count": [net[249], net[9749]],
              "fresh_baseline_mean_cumulative_context": sum(r["baseline_cost"]["context"] for r in rows) / len(rows),
              "fresh_action_mean_cumulative_context": action_cost,
              "posthoc_slack_caps_descriptive_only": descriptive_caps,
              "historical_p0_fixed_schedule_references": references,
              "historical_p0_monotone_fixed_schedules_dominate_fresh_action": len(dominating),
              "limitations": ["Static schedule comparisons use earlier P0 Target outputs and are not fresh paired comparisons.",
                              "All analyses are design-exposed and post-run; they do not override the preregistered M0C opportunity gate."],
              "sealed_sets_read": False}
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
