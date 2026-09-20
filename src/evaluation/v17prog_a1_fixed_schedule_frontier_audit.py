"""Read-only design-side fixed-schedule comparator for PROG-A1 OOF."""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl

LEVELS = (.60, .70, .80, .90, .95)


def main() -> None:
    root = Path("results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
    a1 = Path("results/v2_rank_then_cut/v17prog_a1_state_conditioned_oof")
    source = {r["example_id"]: r for r in read_jsonl(
        "results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/sel_c0_train_top10_candidates.jsonl")
        if fold(r["example_id"]) != 4}
    queries = sorted(source)
    chain = defaultdict(dict)
    for path in sorted(root.glob("shard_*_of_3/per_prefix.jsonl")):
        for row in read_jsonl(path):
            q, depth = row["example_id"], row["depth"]
            if depth in chain[q]:
                raise AssertionError("duplicate P0")
            chain[q][depth] = row
    if len(queries) != 1421 or any(len(chain[q]) != 12 for q in queries):
        raise AssertionError("incomplete train-only P0")
    active = np.array([[level in {float(v) for v in source[q]["attainable_levels"]}
                        for level in LEVELS] for q in queries], dtype=bool)
    f1 = np.array([[chain[q][depth]["f1"] for depth in range(1, 13)] for q in queries])
    tokens = np.array([[chain[q][depth]["tokens"] for depth in range(1, 13)] for q in queries])
    hits = np.stack([f1 + 1e-6 >= level for level in LEVELS], axis=-1) | ~active[:, None, :]
    learned = json.loads((a1 / "summary.json").read_text())["schedules"]["primary"]["learned"]
    target = np.array([learned["success"][str(level)] for level in LEVELS] + [learned["complete"]])
    target_cost = learned["mean_cumulative_context_tokens"]
    dominating = []
    meeting_quality = []
    tested = 0
    for schedule in itertools.combinations_with_replacement(range(1, 13), 5):
        selected = np.stack([hits[:, depth - 1, i] for i, depth in enumerate(schedule)], axis=1)
        quality = np.array([int((selected[:, i] & active[:, i]).sum()) for i in range(5)] +
                           [int(selected.all(axis=1).sum())])
        cost = sum(float((tokens[:, depth - 1] * active[:, i]).mean())
                   for i, depth in enumerate(schedule))
        tested += 1
        if np.all(quality >= target):
            meeting_quality.append((cost, schedule, quality.tolist()))
            if cost <= target_cost + 1e-9:
                dominating.append((cost, schedule, quality.tolist()))
    result = {"role": "post-result read-only design-side diagnostic, not GO gate",
              "schedules_tested": tested, "learned_quality": target.tolist(),
              "learned_mean_cumulative_tokens": target_cost,
              "dominating_v8_schedule_count": len(dominating),
              "cheapest_v8_schedule_matching_all_quality": min(meeting_quality) if meeting_quality else None}
    (a1 / "fixed_schedule_frontier_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
