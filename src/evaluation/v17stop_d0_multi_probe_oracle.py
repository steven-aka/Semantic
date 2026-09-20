"""Costed forced multi-probe oracle using existing train-side prefixes."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.data.schemas import read_jsonl

P0 = Path("results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
C0 = Path("results/v2_rank_then_cut/v17stop_c0_prefix_output_retention")
OUT = Path("results/v2_rank_then_cut/v17stop_d0_multi_probe_oracle")
CONTINUE = "depth10_plus_old_context_supported_depth9"


def main() -> None:
    cfg = json.loads(Path("configs/v17stop_d0_multi_probe_oracle.json").read_text())
    assert cfg["status"] == "FROZEN_READ_ONLY_TRAIN_SIDE"
    rows = {r["example_id"]: r for r in read_jsonl(C0 / "per_query.jsonl")}
    chain = defaultdict(dict)
    for path in sorted(P0.glob("shard_*_of_3/per_prefix.jsonl")):
        for r in read_jsonl(path):
            if r["example_id"] in rows and r["depth"] in (6, 7, 9, 10):
                chain[r["example_id"]][r["depth"]] = r
    assert len(rows) == len(chain) == 1421 and all(len(x) == 4 for x in chain.values())
    report = {"protocol": cfg["protocol"], "population": len(rows),
              "single_depth10": {
                  "success_090": sum(r["outputs"]["depth10_only"]["success"]["0.9"] for r in rows.values()),
                  "mean_target_tokens": sum(r["depth10_target_tokens"] for r in rows.values()) / len(rows)},
              "schedules": {}, "new_target_calls": 0, "sealed_outcome_sets_read": False}
    for depths in cfg["schedules"]:
        assert depths[-2:] == [9, 10]
        total_cost = total_context = total_calls = total_success = 0
        first_success = defaultdict(int)
        for q, r in rows.items():
            for depth in depths:
                state = chain[q][depth]
                total_cost += state["prompt_tokens"] + state["generated_tokens"]
                total_calls += 1
                if depth == 10:
                    success = r["outputs"][CONTINUE]["success"]["0.9"]
                else:
                    success = state["f1"] + 1e-6 >= .9
                if success or depth == 10:
                    total_context += state["tokens"]
                    total_success += success
                    if success:
                        first_success[depth] += 1
                    break
        report["schedules"]["->".join(map(str, depths))] = {
            "success_090": total_success,
            "mean_target_tokens": total_cost / len(rows),
            "mean_final_context_tokens": total_context / len(rows),
            "mean_target_calls": total_calls / len(rows),
            "first_success_depth_counts": dict(first_success)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
