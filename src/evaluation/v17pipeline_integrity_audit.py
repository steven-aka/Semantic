"""Read-only audit of the current train-side experimental lineage and caches."""
from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

from src.evaluation.qampari_metrics import (
    parse_cached_list_prediction, parse_list_prediction, qampari_list_metrics,
)


ROOT = Path("results/v2_rank_then_cut")
LINEAGE = ROOT / "v17sel_b2b_lineage_clean_holdout/data"
OUT = ROOT / "v17pipeline_integrity_audit"


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def ids(path):
    data = [row["example_id"] for row in rows(path)]
    assert len(data) == len(set(data)), path
    return set(data)


def main():
    out = {"scope": "train-side read-only lineage, cached labels and current pilot",
           "target_calls": 0, "checks": {}}
    checks = out["checks"]
    holdout = ids(LINEAGE / "v10_v12_train_holdout.jsonl")
    inner = ids(LINEAGE / "v10_v12_train_inner_validation.jsonl")
    clean = ids(LINEAGE / "v10_v12_train_train_clean.jsonl")
    assert len(holdout) == 581 and len(inner) == 550 and len(clean) == 2032
    assert not (holdout & inner or holdout & clean or inner & clean)
    assert ids(LINEAGE / "v13_train_holdout.jsonl") == holdout
    assert ids(LINEAGE / "v13_train_inner_validation.jsonl") == inner
    assert ids(LINEAGE / "v13_train_train_clean.jsonl") <= clean
    assert ids(LINEAGE / "v8_train_holdout.jsonl") == holdout
    assert ids(LINEAGE / "v8_train_inner_validation.jsonl") == inner
    assert ids(LINEAGE / "v8_train_train_clean.jsonl") == clean
    p0_query = ids(ROOT / "v17canon_p0_fresh_v8_prefix_chain/per_query.jsonl")
    r1_query = ids(ROOT / "v17packet_r1_label_scale512/per_query.jsonl")
    b0_query = ids(ROOT / "v17packet_slot_b0_pilot/per_query.jsonl")
    assert len(p0_query) == 1421 and p0_query <= clean
    assert len(r1_query) == 512 and r1_query <= p0_query
    assert len(b0_query) == 24 and b0_query <= r1_query
    checks["lineage"] = {"holdout": len(holdout), "inner_validation": len(inner),
                         "clean_train": len(clean), "p0": len(p0_query),
                         "r1": len(r1_query), "b0": len(b0_query),
                         "all_disjoint_and_nested": True}

    annotations = {row["example_id"]: row["answer_atoms"] for row in rows(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl")
        if row["example_id"] in p0_query}
    assert len(annotations) == len(p0_query)
    cache_paths = {
        "P0": sorted(glob.glob(str(ROOT / "v17canon_p0_fresh_v8_prefix_chain/shard_*_of_3/per_prefix.jsonl"))),
        "R1": [str(ROOT / "v17packet_r1_label_scale512/per_sentence.jsonl")],
        "SLOT_B0": [str(ROOT / "v17packet_slot_b0_pilot/per_call.jsonl")],
        "EXEC_A1": [str(ROOT / "v17exec_a1_fresh_depth10_cache/per_mask.jsonl")],
    }
    assert len(cache_paths["P0"]) == 3
    caches = {}
    for name, paths in cache_paths.items():
        count = 0
        stored_mismatch = 0
        double_parse_mismatch = 0
        impacted = Counter()
        state_keys = set()
        for path in paths:
            for row in rows(path):
                q = row["example_id"]
                assert q in annotations, (name, q)
                prediction = row["prediction"]
                f1 = float(row["f1"])
                direct = float(qampari_list_metrics(parse_cached_list_prediction(prediction), annotations[q])["f1"])
                twice = float(qampari_list_metrics(parse_list_prediction(prediction), annotations[q])["f1"])
                stored_mismatch += abs(direct - f1) > 1e-8
                if abs(twice - direct) > 1e-8:
                    double_parse_mismatch += 1
                    impacted[q] += 1
                if name == "P0":
                    key = (q, row["depth"])
                    assert key not in state_keys, key
                    state_keys.add(key)
                count += 1
        assert stored_mismatch == 0, (name, stored_mismatch)
        caches[name] = {"rows": count, "stored_f1_mismatch": stored_mismatch,
                        "double_parse_f1_mismatch": double_parse_mismatch,
                        "affected_queries": dict(impacted)}
    assert caches["P0"]["rows"] == 1421 * 12
    assert caches["R1"]["rows"] == 1485
    assert caches["SLOT_B0"]["rows"] == 84
    checks["cache_f1_and_double_parse"] = caches

    b0 = list(rows(ROOT / "v17packet_slot_b0_pilot/per_query.jsonl"))
    assert len(b0) == 24
    complete = lambda hits: all(hit is not False for hit in hits)
    old_complete = lambda hits: all(hits)
    baseline_new = sum(complete(r["baseline_hits"]) for r in b0)
    baseline_old = sum(old_complete(r["baseline_hits"]) for r in b0)
    assert (baseline_old, baseline_new) == (9, 10)
    assert sum(r["baseline_hits"][4] is None for r in b0) == 10
    checks["complete_masking"] = {"four_anchor_queries": 10,
                                  "original_baseline_complete": baseline_old,
                                  "corrected_baseline_complete": baseline_new,
                                  "affected_summaries": ["SLOT-B0", "INSERT-D0"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
