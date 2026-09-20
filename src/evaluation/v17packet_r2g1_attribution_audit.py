"""Read-only R2G0 truncation, fold-scale, and failure-class attribution."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from src.data.schemas import read_jsonl
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import segments


ROOT = Path("results/v2_rank_then_cut")
LINEAGE = ROOT / "v17sel_b2b_lineage_clean_holdout"
R1 = ROOT / "v17packet_r1_label_scale512/per_query.jsonl"
OOF = ROOT / "v17packet_r2g0_relative_bag_probe/oof.jsonl"
CFG = Path("configs/v17packet_r2g0_relative_bag_probe.json")
OUT = ROOT / "v17packet_r2g1_attribution_audit"


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    return sum((x > y) + 0.5 * (x == y) for x in pos for y in neg) / (len(pos)*len(neg))


def summarize(rows: list[dict]) -> dict:
    return {"queries": len(rows), "scored_queries": sum(r["score"] is not None for r in rows),
            "default_action_truncated_queries": sum(r["stay_truncated"] for r in rows),
            "candidate_truncated_at_best_queries": sum(r["best_candidate_truncated"] for r in rows),
            "best_pair_complete_queries": sum(r["best_pair_complete"] for r in rows),
            "repair_queries_with_any_complete_repair": sum(r["any_complete_repair"] for r in rows)}


def main() -> None:
    config = json.loads(CFG.read_text())
    budgets = config["field_token_budgets"]
    rows = list(read_jsonl(R1))
    scores = {x["example_id"]: x for x in read_jsonl(OOF)}
    ids = {r["example_id"] for r in rows}
    source = {x["example_id"]: x for x in read_jsonl(
        LINEAGE / "data/v10_v12_train_train_clean.jsonl") if x["example_id"] in ids}
    orders = {x["example_id"]: x["decoded_order"] for x in read_jsonl(
        LINEAGE / "sel_c0_train_v8_rollouts.jsonl") if x["example_id"] in ids}
    assert len(rows) == len(scores) == len(source) == len(orders) == 512
    tok = AutoTokenizer.from_pretrained(config["encoder_repo"], use_fast=True)
    details = []
    for row in rows:
        q = row["example_id"]
        p = source[q]["packet_texts"]
        order = orders[q]
        parts = segments(p[order[9]])
        assert len(parts) == len(row["sentence_options"])
        stay_len = len(tok.encode(p[order[8]], add_special_tokens=False))
        query_len = len(tok.encode(source[q]["question"], add_special_tokens=False))
        titles = "; ".join(p[i].split("\n", 1)[0] for i in order[:8])
        title_len = len(tok.encode(titles, add_special_tokens=False))
        eligible = []
        complete_repair = 0
        repair = 0
        for action, text in zip(row["sentence_options"], parts):
            if action["context_tokens"] > row["baseline"]["context_tokens"]:
                continue
            candidate_len = len(tok.encode(text, add_special_tokens=False))
            full_pair = stay_len <= budgets["stay_displaced"] and candidate_len <= budgets["candidate"]
            is_repair = row["baseline"]["hits"][3] is False and action["hits"][3] is True
            repair += is_repair
            complete_repair += is_repair and full_pair
            eligible.append({"index": action["index"], "candidate_tokens": candidate_len,
                             "full_pair": full_pair})
        o = scores[q]
        best = next((a for a in eligible if o["best_action"] and
                     a["index"] == o["best_action"]["index"]), None)
        assert (best is None) == (o["best_action"] is None)
        kind = "already_good" if row["baseline"]["hits"][3] else (
            "repairable_failure" if repair else "unrepairable_failure")
        details.append({"example_id": q, "fold": o["fold"], "kind": kind,
                        "score": o["best_score"], "best_action": o["best_action"],
                        "stay_tokens": stay_len, "query_tokens": query_len,
                        "state_title_tokens": title_len,
                        "stay_truncated": stay_len > budgets["stay_displaced"],
                        "query_truncated": query_len > budgets["query"],
                        "titles_truncated": title_len > budgets["state_titles"],
                        "best_candidate_truncated": best is not None and best["candidate_tokens"] > budgets["candidate"],
                        "best_pair_complete": best is not None and best["full_pair"],
                        "any_complete_repair": complete_repair > 0,
                        "repair_action_count": repair,
                        "complete_repair_action_count": complete_repair,
                        "eligible_action_count": len(eligible)})
    groups = {kind: [d for d in details if d["kind"] == kind] for kind in
              ("repairable_failure", "unrepairable_failure", "already_good")}
    assert [len(groups[x]) for x in groups] == [39, 92, 381]
    paired_auc = {}
    for kind in ("unrepairable_failure", "already_good"):
        p = [x["score"] for x in groups["repairable_failure"] if x["score"] is not None]
        n = [x["score"] for x in groups[kind] if x["score"] is not None]
        full_p = [x["score"] for x in groups["repairable_failure"] if x["best_pair_complete"]]
        full_n = [x["score"] for x in groups[kind] if x["best_pair_complete"]]
        paired_auc[f"repairable_vs_{kind}"] = {"raw_pooled_auc": auc(p, n),
            "complete_best_pair_auc": auc(full_p, full_n),
            "complete_best_pair_counts": [len(full_p), len(full_n)],
            "within_fold_auc": {str(f): auc(
                [x["score"] for x in groups["repairable_failure"] if x["fold"] == f and x["score"] is not None],
                [x["score"] for x in groups[kind] if x["fold"] == f and x["score"] is not None])
                for f in range(4)}}
    scored = [d for d in details if d["score"] is not None]
    fold_ranking = {}
    for fraction in (0.05, 0.10):
        selected = []
        by_fold = []
        for f in range(4):
            fold_all = [d for d in details if d["fold"] == f]
            eligible = [d for d in fold_all if d["score"] is not None]
            n = math.ceil(len(fold_all)*fraction)
            pick = sorted(eligible, key=lambda d: (-d["score"], d["example_id"]))[:n]
            selected.extend(pick)
            positives = sum(d["kind"] == "repairable_failure" for d in eligible)
            by_fold.append({"fold": f, "eligible": len(eligible), "selected": len(pick),
                            "repair_opportunities": positives,
                            "found": sum(d["kind"] == "repairable_failure" for d in pick),
                            "random_expected": len(pick)*positives/len(eligible)})
        fold_ranking[str(fraction)] = {"selected": len(selected),
            "opportunities_found": sum(d["kind"] == "repairable_failure" for d in selected),
            "random_expected_found": sum(x["random_expected"] for x in by_fold),
            "repairs": sum(d["kind"] == "repairable_failure" and d["best_action"]["hits"][3] for d in selected),
            "breaks": sum(d["kind"] == "already_good" and not d["best_action"]["hits"][3] for d in selected),
            "selected_with_complete_best_pair": sum(d["best_pair_complete"] for d in selected),
            "folds": by_fold}
    result = {"protocol": "V17-PACKET-R2G1_EXISTING_OUTPUT_ATTRIBUTION_AUDIT",
              "groups": {k: summarize(v) for k,v in groups.items()},
              "score_eligible_queries": len(scored),
              "paired_auc": paired_auc, "within_fold_budgets": fold_ranking,
              "interpretation_limit": "This audits saved R2G0 outputs on design-exposed training queries. Within-fold ranking is diagnostic and cannot be directly deployed as a cross-query score calibration. Complete input visibility does not imply all task-relevant information is encoded.",
              "new_target_calls": 0, "new_training": False, "sealed_sets_read": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUT / "per_query.jsonl").open("w") as handle:
        for d in details:
            d = {k:v for k,v in d.items() if k != "best_action"}
            handle.write(json.dumps(d) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
