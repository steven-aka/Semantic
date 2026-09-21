"""Zero-call, query-clustered audit of cheap pre-Target insertion proxies.

These literal token features are *not* relation entailment or an interference
detector; the point is to see whether even a simple observable signal survives
the very small design-exposed protected-insertion sample.
"""
from __future__ import annotations

import json
import random
import re
from statistics import median
from collections import defaultdict
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import segments


ROOT = Path("results/v2_rank_then_cut")
LINEAGE = ROOT / "v17sel_b2b_lineage_clean_holdout"
B0 = ROOT / "v17packet_slot_b0_pilot"
OUT = ROOT / "v17packet_insert_c0_visible_signal"
STOP = set("a an and are as at be by did do does for from has have how in is it its of on or the to was were what when where which who with whose into than that this their member members list every all".split())


def terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.casefold())
            if len(t) >= 3 and t not in STOP}


def proof(text: str) -> str:
    return text.partition("\nSource evidence: ")[2] or text


def auc(pairs: list[tuple[bool, float]]) -> float | None:
    pos = [v for y, v in pairs if y]
    neg = [v for y, v in pairs if not y]
    if not pos or not neg:
        return None
    return sum((p > n) + .5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def concordance(group: list[dict], feature: str) -> float | None:
    pairs = [(a, b) for i, a in enumerate(group) for b in group[i + 1:]
             if abs(a["delta_f1"] - b["delta_f1"]) > 1e-6]
    if not pairs:
        return None
    return sum((a[feature] - b[feature]) * (a["delta_f1"] - b["delta_f1"]) > 0
               or .5 * ((a[feature] - b[feature]) == 0) for a, b in pairs) / len(pairs)


def query_bootstrap_auc_ci(scores: list[tuple[bool, float]], seed: int = 20260921) -> list[float]:
    rng = random.Random(seed)
    estimates = []
    for _ in range(2000):
        sample = [scores[rng.randrange(len(scores))] for _ in scores]
        value = auc(sample)
        if value is not None:
            estimates.append(value)
    estimates.sort()
    return [estimates[int(.025 * (len(estimates) - 1))],
            estimates[int(.975 * (len(estimates) - 1))]]


def main() -> None:
    baseline = {r["example_id"]: r for r in read_jsonl(B0 / "per_query.jsonl")}
    outputs = list(read_jsonl(B0 / "per_call.jsonl"))
    ids = set(baseline)
    data = {r["example_id"]: r for r in read_jsonl(
        LINEAGE / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(
        LINEAGE / "sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    assert len(ids) == len(data) == len(orders) == 24 and len(outputs) == 84
    records = []
    for row in outputs:
        q = row["example_id"]
        source, order, b = data[q], orders[q], baseline[q]
        question = terms(source["question"])
        selected = [source["packet_texts"][p] for p in order[:9]]
        state = set().union(*(terms(proof(p)) for p in selected))
        candidate = segments(source["packet_texts"][order[9]])[row["candidate_index"]]
        candidate_terms = terms(proof(candidate))
        overlap = len(question & candidate_terms) / max(1, len(question))
        novel_overlap = len((question & candidate_terms) - state) / max(1, len(question))
        packet_terms = [terms(proof(p)) for p in selected]
        redundancy = max((len(candidate_terms & p) / max(1, len(candidate_terms | p))
                          for p in packet_terms), default=0.0)
        bf, af = b["baseline_f1"], row["f1"]
        bh, ah = bf + 1e-6 >= .90, af + 1e-6 >= .90
        record = {"example_id": q, "candidate_index": row["candidate_index"],
                  "baseline_success_090": bh, "action_success_090": ah,
                  "delta_f1": af - bf, "slack": row["slack"],
                  "outcome": "repair" if not bh and ah else "break" if bh and not ah
                  else "same_threshold_f1_gain" if af > bf + 1e-6
                  else "same_threshold_f1_loss" if af < bf - 1e-6 else "unchanged",
                  "query_candidate_overlap": overlap,
                  "novel_query_term_overlap": novel_overlap,
                  "candidate_redundancy": redundancy,
                  "support_novelty_proxy": overlap * (1.0 - redundancy)}
        records.append(record)
    by_query = defaultdict(list)
    for r in records:
        by_query[r["example_id"]].append(r)
    features = ("query_candidate_overlap", "novel_query_term_overlap",
                "candidate_redundancy", "support_novelty_proxy", "slack")
    # Candidate utility = rare threshold repair. Query-level max is a bag
    # score, but positives are selected 12/12 by baseline outcome and not a
    # population prevalence estimate.
    gate = {}
    ranking = {}
    continuous = {}
    for feature in features:
        scores = [(any(r["outcome"] == "repair" for r in group),
                   max(r[feature] for r in group)) for group in by_query.values()]
        gate[feature] = {"query_auc_for_any_repair": auc(scores),
                         "query_bootstrap_95pct_interval": query_bootstrap_auc_ci(scores),
                         "nonzero_action_count": sum(r[feature] > 0 for r in records),
                         "positive_queries": sum(y for y, _ in scores),
                         "negative_queries": sum(not y for y, _ in scores)}
        within = []
        for group in by_query.values():
            positive = [r for r in group if r["outcome"] == "repair"]
            negative = [r for r in group if r["outcome"] != "repair"]
            if positive and negative:
                within.append(auc([(True, r[feature]) for r in positive] +
                                  [(False, r[feature]) for r in negative]))
        ranking[feature] = {"informative_queries": len(within),
                            "macro_within_query_repair_auc": sum(within) / len(within)
                            if within else None}
        within_f1 = [value for group in by_query.values()
                     if (value := concordance(group, feature)) is not None]
        continuous[feature] = {"queries_with_distinct_delta_f1": len(within_f1),
                               "macro_within_query_delta_f1_concordance":
                               sum(within_f1) / len(within_f1) if within_f1 else None}
    outcomes = {name: sum(r["outcome"] == name for r in records)
                for name in ("repair", "break", "same_threshold_f1_gain",
                             "same_threshold_f1_loss", "unchanged")}
    delta_f1 = {name: {"mean": sum(values) / len(values), "median": median(values),
                       "min": min(values), "max": max(values)}
                for name in outcomes if (values := [r["delta_f1"] for r in records
                                                 if r["outcome"] == name])}
    summary = {"protocol": "INSERT-C0_PRETARGET_CHEAP_PROXY_AUDIT",
               "queries": len(by_query), "actions": len(records), "outcomes": outcomes,
               "delta_f1_by_outcome": delta_f1,
               "repair_queries": len({r["example_id"] for r in records if r["outcome"] == "repair"}),
               "break_queries": len({r["example_id"] for r in records if r["outcome"] == "break"}),
               "feature_definitions": {
                   "query_candidate_overlap": "Fraction of non-stopword query tokens literally present in candidate proof.",
                   "novel_query_term_overlap": "Fraction of query tokens present in candidate proof but absent from full V8 depth9 proof text.",
                   "candidate_redundancy": "Maximum Jaccard overlap of candidate proof with any V8 depth9 packet proof; this is not an interference score.",
                   "support_novelty_proxy": "Literal query overlap times (1 - maximum packet redundancy); this does not establish relation support.",
                   "slack": "Actual additional depth9 context tokens, lower is cheaper.",
               },
               "query_bag_diagnostics": gate, "within_query_choice_diagnostics": ranking,
               "continuous_delta_f1_diagnostics": continuous,
               "new_target_calls": 0, "new_training": False, "sealed_sets_read": False,
               "limitations": [
                   "Only 24 baseline-stratified, design-exposed queries; 84 actions are clustered, not independent.",
                   "Only seven queries have any repair and only two queries have a break action; interference separability cannot be assessed reliably.",
                   "These lexical proxies do not operationalize semantic relation entailment or Target-specific interference.",
                   "All features are computed before the depth9 Target call using query, current V8 text and candidate text only; no gold or Target output enters feature computation.",
                   "An AUC near 0.5 here cannot rule out richer deployment-visible signals; an apparent high AUC cannot validate a rule without a new cohort.",
               ]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "per_action.jsonl").open("w") as handle:
        for r in records:
            handle.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
