"""Read-only five-anchor projection and oracle headroom for OBS-A1."""
import glob
import json
from pathlib import Path

BASE = Path("results/v2_rank_then_cut")
OUT = BASE / "v17packet_obs_a1_natural_insertion"
LEVELS = (0.60, 0.70, 0.80, 0.90, 0.95)
SCHEDULE = (6, 7, 7, 9, 10)


def read(path):
    with Path(path).open() as handle:
        for line in handle:
            yield json.loads(line)


def main():
    rows = list(read(OUT / "per_query.jsonl"))
    ids = {r["example_id"] for r in rows}
    data = {r["example_id"]: r for r in read(BASE / "v17sel_b2b_lineage_clean_holdout/data/v10_v12_train_train_clean.jsonl")
            if r["example_id"] in ids}
    cache = {}
    for path in glob.glob(str(BASE / "v17canon_p0_fresh_v8_prefix_chain/shard_*_of_3/per_prefix.jsonl")):
        for row in read(path):
            if row["example_id"] in ids and row["depth"] in (6, 7, 10):
                cache[row["example_id"], row["depth"]] = row
    assert len(rows) == len(data) == 256
    assert len(cache) == 256 * 3
    per_query = []
    for row in rows:
        q = row["example_id"]
        attainable = data[q]["attainable_levels"]
        assert 0.9 in attainable or "0.9" in attainable
        other = []
        for i, (level, depth) in enumerate(zip(LEVELS, SCHEDULE)):
            if i == 3:
                other.append(None)
            elif level not in attainable and str(level) not in attainable:
                other.append(None)
            else:
                other.append(cache[q, depth]["f1"] + 1e-6 >= level)
        baseline = other[:]
        action = other[:]
        baseline[3] = row["baseline_f1"] + 1e-6 >= .9
        action[3] = row["action_f1"] + 1e-6 >= .9
        before, after = all(v is not False for v in baseline), all(v is not False for v in action)
        per_query.append({"example_id": q, "baseline_hits": baseline, "action_hits": action,
                          "baseline_complete": before, "action_complete": after,
                          "complete_repair": not before and after,
                          "complete_break": before and not after,
                          "other_anchors_all_pass": all(v is not False for i, v in enumerate(other) if i != 3),
                          "repair_090": row["repair"], "break_090": row["break"],
                          "slack": row["slack"], "target_token_delta": row["target_token_delta"]})
    repair = [r for r in per_query if r["repair_090"]]
    complete_repair = [r for r in per_query if r["complete_repair"]]
    result = {
        "protocol": "OBS-A3_FIVE_ANCHOR_HYBRID_PROJECTION",
        "schedule": list(SCHEDULE), "queries": len(per_query),
        "baseline_five_anchor_success": [sum(r["baseline_hits"][i] is True for r in per_query) for i in range(5)],
        "action_five_anchor_success": [sum(r["action_hits"][i] is True for r in per_query) for i in range(5)],
        "baseline_complete": sum(r["baseline_complete"] for r in per_query),
        "action_complete": sum(r["action_complete"] for r in per_query),
        "complete_repairs": sum(r["complete_repair"] for r in per_query),
        "complete_breaks": sum(r["complete_break"] for r in per_query),
        "090_repairs_with_all_other_anchors_passing": sum(r["other_anchors_all_pass"] for r in repair),
        "oracle_repair_only_090": sum(r["baseline_hits"][3] is True for r in per_query) + len(repair),
        "oracle_repair_only_complete": sum(r["baseline_complete"] for r in per_query) + sum(r["complete_repair"] for r in per_query),
        "oracle_repair_only_mean_extra_context_tokens": sum(r["slack"] for r in repair) / len(per_query),
        "oracle_repair_only_mean_extra_target_tokens": sum(r["target_token_delta"] for r in repair) / len(per_query),
        "oracle_complete_repair_only_mean_extra_context_tokens": sum(r["slack"] for r in complete_repair) / len(per_query),
        "oracle_complete_repair_only_mean_extra_target_tokens": sum(r["target_token_delta"] for r in complete_repair) / len(per_query),
        "sealed_sets_read": False,
        "limitation": "0.90 comes from fresh paired OBS-A1 calls; other unchanged anchors come from earlier P0 runs. This is a hybrid projection and hindsight oracle, not a freshly paired end-to-end result."
    }
    (OUT / "complete_bound.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUT / "complete_bound_per_query.jsonl").open("w") as handle:
        for row in per_query:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
