"""Train-side grouped OOF diagnostic for cheap answer-count stopping signal."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import normalize_list_answer, parse_list_prediction
from src.evaluation.v17stop_v1_oracle_probe_cost import LEVELS, ROOT, call_cost

OUT = Path("results/v2_rank_then_cut/v17stop_v1_answer_count_oof")


def answer_count(prediction: str) -> int:
    return len({normalize_list_answer(x) for x in parse_list_prediction(prediction)
                if normalize_list_answer(x)})


def main() -> None:
    cfg = json.loads(Path("configs/v17stop_v1_answer_count_oof.json").read_text())
    if cfg["status"] != "FROZEN_TRAIN_SIDE_DIAGNOSTIC":
        raise AssertionError("unfrozen protocol")
    source = {r["example_id"]: r for r in read_jsonl(
        "results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/sel_c0_train_top10_candidates.jsonl")
        if fold(r["example_id"]) != 4}
    chains = defaultdict(dict)
    for path in sorted(ROOT.glob("shard_*_of_3/per_prefix.jsonl")):
        for row in read_jsonl(path):
            chains[row["example_id"]][row["depth"]] = row
    if len(source) != 1421 or set(source) != set(chains) or any(len(v) != 12 for v in chains.values()):
        raise AssertionError("fresh prefix cache mismatch")
    counts = {q: {depth: answer_count(chains[q][depth]["prediction"])
                  for depth in cfg["probe_depths"] if depth < 10} for q in source}
    # The lineage split has five values; fold 4 is the 611-query sealed gate.
    buckets = {q: fold(q) for q in source}
    if set(buckets.values()) != {0, 1, 2, 3}:
        raise AssertionError("unexpected OOF split")
    thresholds = {}
    for heldout in range(4):
        thresholds[heldout] = {}
        train = [q for q in source if buckets[q] != heldout]
        for i, (level, depth) in enumerate(zip(LEVELS, cfg["probe_depths"])):
            if depth == 10:
                thresholds[heldout][str(level)] = None
                continue
            active = [q for q in train if level in {float(x) for x in source[q]["attainable_levels"]}]
            feasible = []
            for threshold in range(1, 21):
                chosen = [q for q in active if counts[q][depth] >= threshold]
                breaks = sum(chains[q][10]["f1"] + 1e-6 >= level and
                             chains[q][depth]["f1"] + 1e-6 < level for q in chosen)
                if len(chosen) >= 20 and breaks == 0:
                    saving = sum(chains[q][10]["tokens"] - chains[q][depth]["tokens"] for q in chosen)
                    feasible.append((saving, threshold))
            thresholds[heldout][str(level)] = max(feasible)[1] if feasible else None
    rows = []
    per_anchor = defaultdict(lambda: defaultdict(int))
    for query, metadata in source.items():
        active = {float(x) for x in metadata["attainable_levels"]}
        bucket = buckets[query]
        base_context = policy_context = base_compute = policy_compute = 0
        base_complete = policy_complete = True
        base_success = policy_success = 0
        calls = early = 0
        for level, depth in zip(LEVELS, cfg["probe_depths"]):
            if level not in active:
                continue
            base = chains[query][10]
            probe = chains[query][depth]
            threshold = thresholds[bucket][str(level)]
            probe_enabled = depth < 10 and threshold is not None
            stop = probe_enabled and counts[query][depth] >= threshold
            final = probe if stop else base
            # A depth-10 query is the baseline call. Earlier probes happen
            # regardless of the decision and require fallback when continuing.
            policy_compute += call_cost(probe) if probe_enabled else call_cost(base)
            calls += 1
            if probe_enabled and not stop:
                policy_compute += call_cost(base)
                calls += 1
            early += stop
            base_context += base["tokens"]
            policy_context += final["tokens"]
            base_compute += call_cost(base)
            b_ok = base["f1"] + 1e-6 >= level
            p_ok = final["f1"] + 1e-6 >= level
            base_complete &= b_ok
            policy_complete &= p_ok
            base_success += b_ok
            policy_success += p_ok
            k = str(level)
            per_anchor[k]["requests"] += 1
            per_anchor[k]["early_exits"] += stop
            per_anchor[k]["repairs"] += not b_ok and p_ok
            per_anchor[k]["breaks"] += b_ok and not p_ok
        rows.append({"example_id": query, "fold": bucket, "early_exits": early,
                     "calls": calls, "base_context": base_context,
                     "policy_context": policy_context, "base_compute": base_compute,
                     "policy_compute": policy_compute, "base_complete": base_complete,
                     "policy_complete": policy_complete, "base_success": base_success,
                     "policy_success": policy_success})
    report = {"protocol": cfg["protocol"], "status": cfg["status"],
              "feature": cfg["decision_feature"], "thresholds_by_oof_fold": thresholds,
              "queries": len(rows), "per_anchor": dict(per_anchor),
              "baseline_anchor_success_sum": sum(r["base_success"] for r in rows),
              "policy_anchor_success_sum": sum(r["policy_success"] for r in rows),
              "baseline_complete": sum(r["base_complete"] for r in rows),
              "policy_complete": sum(r["policy_complete"] for r in rows),
              "mean_baseline_final_context_tokens": mean(r["base_context"] for r in rows),
              "mean_policy_final_context_tokens": mean(r["policy_context"] for r in rows),
              "mean_baseline_target_compute_tokens": mean(r["base_compute"] for r in rows),
              "mean_policy_target_compute_tokens": mean(r["policy_compute"] for r in rows),
              "mean_extra_target_calls": mean(r["calls"] - len(source[r["example_id"]]["attainable_levels"]) for r in rows),
              "new_target_calls": 0,
              "sealed_sets_read": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUT / "per_query.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
