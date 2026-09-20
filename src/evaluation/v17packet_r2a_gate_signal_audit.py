"""Frozen-score opportunity detection audit; no new inference or Target calls."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path("results/v2_rank_then_cut")
R1 = ROOT / "v17packet_r1_label_scale512/per_query.jsonl"
R2 = ROOT / "v17packet_r2_semantic_probe/oof.jsonl"
OUT = ROOT / "v17packet_r2a_gate_signal_audit"
BUDGETS = (0.01, 0.05, 0.10, 0.15)


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def main():
    rows = read(R1)
    scores = {x["example_id"]: x for x in read(R2)}
    assert len(rows) == len(scores) == 512
    indexed = []
    for row in rows:
        q = row["example_id"]
        b = row["baseline"]
        o = scores[q]
        repairable = any(
            a["context_tokens"] <= b["context_tokens"] and
            b["hits"][3] is False and a["hits"][3] is True
            for a in row["sentence_options"]
        )
        assert b == o["baseline"]
        indexed.append({"example_id": q, "fold": o["fold"],
                        "baseline_success": b["hits"][3], "repairable": repairable,
                        "score": o["best_score"], "best_action": o["best_action"]})
    assert sum(x["repairable"] for x in indexed) == 39
    ranked = sorted([x for x in indexed if x["score"] is not None],
                    key=lambda x: (-x["score"], x["example_id"]))
    eligible_n = len(ranked)
    eligible_positive_n = sum(x["repairable"] for x in ranked)
    assert eligible_positive_n == 39
    def report(n):
        selected = ranked[:n]
        k = sum(x["repairable"] for x in selected)
        repairs = sum(not x["baseline_success"] and x["best_action"]["hits"][3] for x in selected)
        breaks = sum(x["baseline_success"] and not x["best_action"]["hits"][3] for x in selected)
        # Random ranking is restricted to the same score-eligible pool.
        denom = math.comb(eligible_n, n)
        p_upper = sum(math.comb(eligible_positive_n, j) *
                      math.comb(eligible_n-eligible_positive_n, n-j)
                      for j in range(k, min(eligible_positive_n,n)+1)
                      if 0 <= n-j <= eligible_n-eligible_positive_n) / denom
        return {"switch_budget_queries": n, "opportunities_found": k,
                "opportunity_recall": k/39, "opportunity_precision": k/n,
                "random_expected_opportunities": n*eligible_positive_n/eligible_n,
                "random_enrichment_ratio": k/(n*eligible_positive_n/eligible_n),
                "hypergeometric_p_enrichment": p_upper,
                "actual_repairs": repairs, "actual_breaks": breaks,
                "baseline_failures_selected": sum(not x["baseline_success"] for x in selected)}
    budgets = {str(f): report(math.ceil(512*f)) for f in BUDGETS}
    result = {
        "protocol": "V17-PACKET-R2A_FROZEN_SCORE_GATE_SIGNAL_AUDIT",
        "queries": 512, "baseline_success_queries": 381,
        "baseline_failure_queries": 131, "repairable_failure_queries": 39,
        "unrepairable_failure_queries": 92,
        "eligible_scored_queries": len(ranked),
        "random_baseline_population": "422 score-eligible queries, including all 39 repairable failures",
        "budgets": budgets,
        "decision": "NO_ENRICHMENT_FROM_SAVED_R2_TOP_SCORE",
        "interpretation": "This rejects the saved absolute-success top-score as a useful low-budget repair gate on these exposed queries. It does not test candidate-bag features or a paired model: R2 did not save all candidate scores or checkpoints.",
        "new_target_calls": 0, "new_training": False,
        "sealed_sets_read": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
