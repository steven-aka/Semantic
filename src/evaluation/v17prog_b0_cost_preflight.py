"""Count fresh Target states needed by bounded remaining-packet moves."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl


def main() -> None:
    cfg = json.loads(Path("configs/v17prog_b0_cost_preflight.json").read_text())
    if cfg["status"] != "COST_PREFLIGHT_ONLY_NO_TARGET_CALLS":
        raise ValueError("preflight only")
    source = {r["example_id"] for r in read_jsonl(
        "results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/sel_c0_train_top10_candidates.jsonl")
        if fold(r["example_id"]) != 4}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(
        "results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/sel_c0_train_v8_rollouts.jsonl")
        if r["example_id"] in source}
    existing = defaultdict(set)
    root = Path("results/v2_rank_then_cut")
    for path in sorted((root / "v17canon_p0_fresh_v8_prefix_chain").glob("shard_*_of_3/per_prefix.jsonl")):
        for r in read_jsonl(path):
            existing[r["example_id"]].add(r["mask"])
    for r in read_jsonl(root / "v17prog_a0_fresh_adjacent_swap_oracle/per_action.jsonl"):
        existing[r["example_id"]].add(r["mask"])
    if len(source) != 1421 or len(orders) != 1421 or len(existing) != 1421:
        raise AssertionError("missing current-contract source")
    stops = {d for schedule in cfg["fixed_schedules"] for d in schedule}
    scenarios = [cfg["stage1_move_depths"], *cfg["conditional_extension_move_depths"]]
    result = {"protocol": cfg["protocol"], "queries": 1421, "new_target_calls": 0,
              "fixed_stop_depths": sorted(stops), "scenarios": {}}
    for depths in scenarios:
        new_states = actions = 0
        for q in sorted(source):
            order = orders[q]
            needed = set()
            for depth in depths:
                for rank in range(depth + 1, 13):
                    moved = list(order)
                    moved.insert(depth - 1, moved.pop(rank - 1))
                    actions += 1
                    for stop in stops:
                        mask = sum(1 << packet for packet in moved[:stop])
                        if mask not in existing[q]:
                            needed.add(mask)
            new_states += len(needed)
        result["scenarios"][",".join(map(str, depths))] = {
            "candidate_move_actions_including_cached_adjacent": actions,
            "deduplicated_new_target_states": new_states,
            "average_new_states_per_query": new_states / len(source)}
    out = root / "v17prog_b0_cost_preflight"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
