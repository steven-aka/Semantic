"""Map attainable-anchor failures on the frozen single V8 trajectory."""
from collections import Counter
import json
from pathlib import Path

from src.data.schemas import read_jsonl

BASE = Path("results/v2_rank_then_cut")
OUT = BASE / "v17traj_m0a_failure_locus"
LEVELS = (0.60, 0.70, 0.80, 0.90, 0.95)
SCHEDULE = (6, 7, 7, 9, 10)


def main():
    cfg = json.loads(Path("configs/v17canon_p0_fresh_v8_prefix_chain.json").read_text())
    p0 = BASE / "v17canon_p0_fresh_v8_prefix_chain"
    source = {r["example_id"]: {float(level) for level in r["attainable_levels"]}
              for r in read_jsonl(BASE / "v17sel_b2b_lineage_clean_holdout/data/v10_v12_train_train_clean.jsonl")}
    required_depths = set(SCHEDULE)
    chains = {}
    for path in sorted(p0.glob("shard_*_of_3/per_prefix.jsonl")):
        for row in read_jsonl(path):
            if row["depth"] in required_depths:
                key = (row["example_id"], row["depth"])
                assert key not in chains
                chains[key] = row
    ids = {r["example_id"] for r in read_jsonl(p0 / "per_query.jsonl")}
    assert len(ids) == 1421 and len(chains) == len(ids) * len(required_depths)
    masks = Counter()
    failed = Counter()
    pairs = Counter()
    per_query = []
    for q in sorted(ids):
        assert q in source
        hits = []
        mask = []
        for level, depth in zip(LEVELS, SCHEDULE):
            if level not in source[q]:
                hits.append(None)
                mask.append("-")
            else:
                success = chains[q, depth]["f1"] + cfg["fidelity_epsilon"] >= level
                hits.append(success)
                mask.append("1" if success else "0")
                if not success:
                    failed[str(level)] += 1
        key = "".join(mask)
        masks[key] += 1
        for i in range(len(LEVELS)):
            for j in range(i + 1, len(LEVELS)):
                if hits[i] is False and hits[j] is False:
                    pairs[f"{LEVELS[i]}+{LEVELS[j]}"] += 1
        per_query.append({"example_id": q, "success_vector": hits,
                          "failure_mask": key, "complete": all(v is not False for v in hits)})
    assert sum(masks.values()) == len(ids)
    assert sum(r["complete"] for r in per_query) == 775
    result = {"protocol": "TRAJ-M0A_FROZEN_V8_FAILURE_LOCUS",
              "schedule": list(SCHEDULE), "queries": len(ids), "complete": 775,
              "noncomplete": len(ids) - 775, "mask_counts": dict(masks.most_common()),
              "failed_by_anchor": dict(failed), "pairwise_failure_cooccurrence": dict(pairs),
              "single_anchor_failure": {str(level): sum(
                  sum(v is False for v in r["success_vector"]) == 1 and r["success_vector"][i] is False
                  for r in per_query) for i, level in enumerate(LEVELS)},
              "sealed_sets_read": False,
              "interpretation": "Failure masks are observational diagnostics on one fixed V8 trajectory; they do not prove that one action can repair multiple failures."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUT / "per_query.jsonl").open("w") as handle:
        for row in per_query:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
