"""Retrospective gold-alias eviction audit on A3 train-side outcomes."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.v17packet_mech_a0_cached_audit import gold_mentions, matched_atoms
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render, segments


ROOT = Path("results/v2_rank_then_cut")
LINEAGE = ROOT / "v17sel_b2b_lineage_clean_holdout"
P0 = ROOT / "v17canon_p0_fresh_v8_prefix_chain"
A3 = ROOT / "v17packet_mech_a3_success_safety"
OUT = ROOT / "v17packet_evict_a0_gold_diagnostic"


def main() -> None:
    actions = list(read_jsonl(A3 / "per_action.jsonl"))
    ids = {r["example_id"] for r in actions}
    source = {r["example_id"]: r for r in read_jsonl(
        LINEAGE / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(
        LINEAGE / "sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    cache = {}
    for path in sorted(P0.glob("shard_*_of_3/per_prefix.jsonl")):
        for row in read_jsonl(path):
            if row["example_id"] in ids:
                cache[row["example_id"],row["mask"]] = row
    assert len(actions) == 153 and len(ids) == len(source) == len(orders) == len(atoms) == 32
    details = []
    for action in actions:
        q = action["example_id"]
        packets = source[q]["packet_texts"]
        order = orders[q]
        omitted = segments(packets[order[8]])[action["omit_index"]]
        remaining = [x for i,x in enumerate(segments(packets[order[8]]))
                     if i != action["omit_index"]]
        remaining_context = render(packets,mask(order,8)) + "\n\n" + " ".join(remaining)
        promoted = segments(packets[order[9]])[action["candidate_index"]]
        baseline = cache[q,mask(order,9)]
        baseline_answers = matched_atoms(baseline["prediction"],atoms[q])
        action_answers = matched_atoms(action["prediction"],atoms[q])
        omitted_mentions = gold_mentions(omitted,atoms[q])
        unique_mentions = omitted_mentions - gold_mentions(remaining_context,atoms[q])
        lost = baseline_answers-action_answers
        details.append({"example_id": q, "omit_index": action["omit_index"],
                        "candidate_index": action["candidate_index"],
                        "breaks_090": action["f1"]+1e-6 < .90,
                        "omitted_mentions_gold": bool(omitted_mentions),
                        "omitted_mentions_baseline_answer": bool(omitted_mentions & baseline_answers),
                        "unique_gold_mention": bool(unique_mentions),
                        "unique_gold_not_replaced_by_promoted": bool(unique_mentions-gold_mentions(promoted,atoms[q])),
                        "lost_answer_matches_omitted": bool(lost & omitted_mentions),
                        "baseline_correct_atoms_lost": len(lost)})
    summary = {"protocol": "V17-PACKET-EVICT-A0_GOLD_ALIAS_DIAGNOSTIC",
               "actions": len(details), "queries": len(ids),
               "break_actions": sum(r["breaks_090"] for r in details),
               "queries_with_any_break_action": len({r["example_id"] for r in details if r["breaks_090"]}),
               "crosstab": {},
               "limitations": [
                   "Gold alias mention is not proof of query relation support; absence is not proof of no support.",
                   "Actions within a query are correlated; action counts are not independent samples.",
                   "Lost-answer correspondence is a retrospective outcome diagnostic, unavailable to a deployment controller.",
                   "The V8 rank9 sentence is deferred relative to V8, not removed from already revealed depth8 context."
               ], "new_target_calls": 0, "new_training": False, "sealed_sets_read": False}
    for key in ("omitted_mentions_gold", "omitted_mentions_baseline_answer",
                "unique_gold_mention", "unique_gold_not_replaced_by_promoted",
                "lost_answer_matches_omitted"):
        summary["crosstab"][key] = {
            "break_true": sum(r["breaks_090"] and r[key] for r in details),
            "break_false": sum(r["breaks_090"] and not r[key] for r in details),
            "safe_true": sum(not r["breaks_090"] and r[key] for r in details),
            "safe_false": sum(not r["breaks_090"] and not r[key] for r in details)}
    protected = defaultdict(list)
    for r in details:
        if not r["omitted_mentions_baseline_answer"]:
            protected[r["example_id"]].append(r)
    summary["gold_alias_filter"] = {
        "remaining_actions": sum(map(len,protected.values())),
        "remaining_queries": len(protected),
        "remaining_breaks": sum(r["breaks_090"] for v in protected.values() for r in v)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "per_action.jsonl").open("w") as handle:
        for r in details:
            handle.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
