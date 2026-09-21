"""Locate historical V8 depth10-to-12 0.95 rollbacks without Target calls."""
import glob
import json
from collections import defaultdict
from pathlib import Path

from src.data.schemas import read_jsonl

BASE = Path("results/v2_rank_then_cut")
OUT = BASE / "v17traj_m2_095_locus_audit"


def main():
    allowed = {r["example_id"] for r in read_jsonl(BASE / "v17sel_b2b_lineage_clean_holdout/data/v10_v12_train_train_clean.jsonl")
               if 0.95 in r["attainable_levels"]}
    prefixes = defaultdict(dict)
    for filename in glob.glob(str(BASE / "v17canon_p0_fresh_v8_prefix_chain/shard_*_of_3/per_prefix.jsonl")):
        for r in read_jsonl(filename):
            if r["example_id"] in allowed and r["depth"] in (10, 11, 12):
                prefixes[r["example_id"]][r["depth"]] = r["f1"]
    assert len(prefixes) == 1131 and all(set(v) == {10, 11, 12} for v in prefixes.values())
    rows = []
    for q, v in sorted(prefixes.items()):
        success = {d: v[d] + 1e-6 >= 0.95 for d in (10, 11, 12)}
        if success[10] and not success[12]:
            rows.append({"example_id": q, "f1_at_depth10": v[10], "f1_at_depth11": v[11],
                         "f1_at_depth12": v[12], "first_failure_transition": "10_to_11" if not success[11] else "11_to_12"})
    result = {"protocol": "TRAJ_M2_095_ROLLBACK_LOCUS_ZERO_CALL", "attainable_095_queries": len(prefixes),
              "depth10_success_depth12_failure": len(rows),
              "first_failure_at_10_to_11": sum(r["first_failure_transition"] == "10_to_11" for r in rows),
              "first_failure_at_11_to_12": sum(r["first_failure_transition"] == "11_to_12" for r in rows),
              "new_target_calls": 0, "sealed_sets_read": False,
              "limitation": "This localizes the observed transition only; it does not identify a packet-level causal mechanism."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUT / "rollback_queries.jsonl").open("w") as handle:
        for r in rows:
            handle.write(json.dumps(r) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
