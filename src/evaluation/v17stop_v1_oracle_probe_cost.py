"""Costed omniscient single-probe stopping ceiling on fresh train-side prefixes."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl

LEVELS = (.60, .70, .80, .90, .95)
ROOT = Path("results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
OUT = Path("results/v2_rank_then_cut/v17stop_v1_oracle_probe_cost")


def call_cost(row: dict) -> int:
    return row["prompt_tokens"] + row["generated_tokens"]


def evaluate(source: dict, chains: dict, depths: list[int], sequential: bool) -> dict:
    rows = []
    for query, metadata in source.items():
        active = {float(level) for level in metadata["attainable_levels"]}
        chain = chains[query]
        baseline_context = baseline_compute = probe_context = probe_compute = 0
        baseline_success = probe_success = 0
        baseline_complete = probe_complete = True
        calls = 0
        previous = 0
        for level, nominal_depth in zip(LEVELS, depths):
            if level not in active:
                continue
            base = chain[10]
            b_ok = base["f1"] + 1e-6 >= level
            baseline_success += b_ok
            baseline_complete &= b_ok
            baseline_context += base["tokens"]
            baseline_compute += call_cost(base)
            depth = max(previous, nominal_depth) if sequential else nominal_depth
            probe = chain[depth]
            calls += 1
            probe_compute += call_cost(probe)
            if probe["f1"] + 1e-6 >= level or depth == 10:
                final = probe
            else:
                final = base
                probe_compute += call_cost(base)
                calls += 1
            previous = 10 if final is base else depth
            p_ok = final["f1"] + 1e-6 >= level
            probe_success += p_ok
            probe_complete &= p_ok
            probe_context += final["tokens"]
        rows.append({"example_id": query, "baseline_context": baseline_context,
                     "probe_context": probe_context, "baseline_compute": baseline_compute,
                     "probe_compute": probe_compute, "calls": calls,
                     "baseline_success": baseline_success, "probe_success": probe_success,
                     "baseline_complete": baseline_complete,
                     "probe_complete": probe_complete})
    return {
        "queries": len(rows), "attainable_requests": sum(len(source[q]["attainable_levels"]) for q in source),
        "baseline_anchor_success_sum": sum(r["baseline_success"] for r in rows),
        "oracle_probe_anchor_success_sum": sum(r["probe_success"] for r in rows),
        "baseline_complete": sum(r["baseline_complete"] for r in rows),
        "oracle_probe_complete": sum(r["probe_complete"] for r in rows),
        "mean_baseline_final_context_tokens": mean(r["baseline_context"] for r in rows),
        "mean_probe_final_context_tokens": mean(r["probe_context"] for r in rows),
        "mean_final_context_tokens_saved": mean(r["baseline_context"] - r["probe_context"] for r in rows),
        "mean_baseline_target_compute_tokens": mean(r["baseline_compute"] for r in rows),
        "mean_probe_target_compute_tokens": mean(r["probe_compute"] for r in rows),
        "mean_target_compute_tokens_saved": mean(r["baseline_compute"] - r["probe_compute"] for r in rows),
        "mean_extra_target_calls": mean(r["calls"] - len(source[r["example_id"]]["attainable_levels"]) for r in rows),
        "queries_with_target_compute_saving": sum(r["probe_compute"] < r["baseline_compute"] for r in rows),
    }


def main() -> None:
    cfg = json.loads(Path("configs/v17stop_v1_oracle_probe_cost.json").read_text())
    if cfg["status"] != "FROZEN_READ_ONLY_TRAIN_SIDE":
        raise AssertionError("protocol not frozen")
    source = {r["example_id"]: r for r in read_jsonl(
        "results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/sel_c0_train_top10_candidates.jsonl")
        if fold(r["example_id"]) != 4}
    chains = defaultdict(dict)
    for path in sorted(ROOT.glob("shard_*_of_3/per_prefix.jsonl")):
        for row in read_jsonl(path):
            if row["depth"] in chains[row["example_id"]]:
                raise AssertionError("duplicate prefix")
            chains[row["example_id"]][row["depth"]] = row
    if len(source) != 1421 or set(source) != set(chains) or any(len(v) != 12 for v in chains.values()):
        raise AssertionError("incomplete or mismatched fresh cache")
    report = {"protocol": cfg["protocol"], "status": cfg["status"], "rules": cfg["rules"],
              "schedules": {}, "new_target_calls": 0}
    for name, depths in cfg["probe_schedules"].items():
        if len(depths) != len(LEVELS) or sorted(depths) != depths or depths[-1] != 10:
            raise AssertionError("invalid schedule")
        report["schedules"][name] = {"depths": depths,
            "independent_requests": evaluate(source, chains, depths, False),
            "sequential_monotone": evaluate(source, chains, depths, True)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["schedules"], indent=2))


if __name__ == "__main__":
    main()
