"""Audit the fresh V8 prefix chain without fitting or choosing a controller."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl

LEVELS = (0.60, 0.70, 0.80, 0.90, 0.95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v17canon_p0_fresh_v8_prefix_chain.json")
    parser.add_argument("--root", default="results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
    parser.add_argument("--candidates", default="results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/sel_c0_train_top10_candidates.jsonl")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    source = {r["example_id"]: r for r in read_jsonl(args.candidates) if fold(r["example_id"]) != 4}
    if len(source) != 1421:
        raise AssertionError("unexpected population")
    root = Path(args.root)
    paths = sorted(root.glob("shard_*_of_*/per_prefix.jsonl"))
    if not paths:
        raise ValueError("no prefix records")
    chains = defaultdict(dict)
    for path in paths:
        for row in read_jsonl(path):
            q, d = row["example_id"], row["depth"]
            if q not in source or d in chains[q]:
                raise AssertionError("unexpected or duplicate prefix")
            chains[q][d] = row
    if len(chains) != 1421 or any(set(c) != set(range(1, 13)) for c in chains.values()):
        raise AssertionError("incomplete fresh prefix chain; do not report results")
    active = {q: {float(x) for x in row["attainable_levels"]} for q, row in source.items()}
    def hit(q, d, level):
        return level not in active[q] or chains[q][d]["f1"] + cfg["fidelity_epsilon"] >= level
    by_depth = {}
    for d in range(1, 13):
        by_depth[str(d)] = {
            "eligible": {str(l): sum(l in active[q] for q in chains) for l in LEVELS},
            "success": {str(l): sum(l in active[q] and hit(q, d, l) for q in chains) for l in LEVELS},
            "complete_all_attainable_at_this_depth": sum(all(hit(q, d, l) for l in LEVELS) for q in chains),
            "mean_context_tokens": mean(chains[q][d]["tokens"] for q in chains),
            "mean_context_fraction": mean(chains[q][d]["tokens"] / chains[q][d]["full_tokens"] for q in chains),
        }
    per_query = []
    for q in sorted(chains):
        levels = {}
        for level in LEVELS:
            if level not in active[q]:
                levels[str(level)] = None
                continue
            successful = [d for d in range(1, 13) if hit(q, d, level)]
            first = successful[0] if successful else None
            width = 0
            if first is not None:
                for d in range(first, 13):
                    if not hit(q, d, level):
                        break
                    width += 1
            levels[str(level)] = {"earliest_success": first, "first_window_width": width,
                                  "rollback_edges": [d for d in range(1, 12) if hit(q, d, level) and not hit(q, d + 1, level)]}
        per_query.append({"example_id": q, "levels": levels})
    out = {"protocol": cfg["protocol"], "queries": len(chains), "prefix_states": sum(map(len, chains.values())),
           "reused_depth10": sum(chains[q][10]["source"].startswith("EXEC-A1") for q in chains),
           "target_calls_generated": sum(row["source"] == "CANON-P0 current-contract" for c in chains.values() for row in c.values()),
           "prompt_tokens_generated": sum(row["prompt_tokens"] for c in chains.values() for row in c.values() if row["source"] == "CANON-P0 current-contract"),
           "generated_tokens_generated": sum(row["generated_tokens"] for c in chains.values() for row in c.values() if row["source"] == "CANON-P0 current-contract"),
           "fixed_single_depth": by_depth,
           "interpretation": "Each depth applies the same stop to all five anchors. These are diagnostic operating points, not a level-wise deployed schedule or a learned policy."}
    (root / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    with (root / "per_query.jsonl").open("w") as handle:
        for row in per_query:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"queries": len(chains), "prefix_states": out["prefix_states"],
                      "depth10": by_depth["10"]}), flush=True)


if __name__ == "__main__":
    main()
