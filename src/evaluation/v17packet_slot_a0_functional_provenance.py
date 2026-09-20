"""Train-side cached answer-transition audit for depth-9 partial swaps.

Answer changes are observational; simultaneous deferral and addition cannot
identify the causal contribution of either component.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.v17packet_mech_a0_cached_audit import matched_atoms, normalized_answers
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask


ROOT = Path("results/v2_rank_then_cut")
LINEAGE = ROOT / "v17sel_b2b_lineage_clean_holdout"
P0 = ROOT / "v17canon_p0_fresh_v8_prefix_chain"
A2 = ROOT / "v17packet_mech_a2_budget_neutral_swap"
A3 = ROOT / "v17packet_mech_a3_success_safety"
OUT = ROOT / "v17packet_slot_a0_functional_provenance"
EPS = 1e-6


def transition(before: dict, baseline: dict, alternative: dict, atoms: list[dict]) -> dict:
    a8 = matched_atoms(before["prediction"], atoms)
    b9 = matched_atoms(baseline["prediction"], atoms)
    s9 = matched_atoms(alternative["prediction"], atoms)
    lost = b9 - s9
    gained = s9 - b9
    early_lost = lost & a8
    late_lost = lost - a8
    b_false = len(normalized_answers(baseline["prediction"])) - len(b9)
    s_false = len(normalized_answers(alternative["prediction"])) - len(s9)
    return {
        "depth8_correct": len(a8), "baseline_correct": len(b9),
        "action_correct": len(s9), "lost_correct": len(lost),
        "early_lost": len(early_lost), "late_lost": len(late_lost),
        "new_correct": len(gained), "false_answer_delta": s_false - b_false,
        "depth8_f1": before["f1"], "baseline_f1": baseline["f1"],
        "action_f1": alternative["f1"],
        "baseline_margin_090": baseline["f1"] - .90,
        "action_margin_090": alternative["f1"] - .90,
    }


def main() -> None:
    actions = [("safety", r) for r in read_jsonl(A3 / "per_action.jsonl")]
    stable = {r["example_id"] for r in read_jsonl(A2 / "per_case.jsonl")
              if r["group"] == "repair" and r["prior_repeat_pair_valid"]
              and r["oracle_success"]}
    repair_actions = [r for r in read_jsonl(A2 / "per_action.jsonl")
                      if r["example_id"] in stable and r["f1"] + EPS >= .90]
    # One retrospectively cheapest successful action per stable repair case.
    selected = {}
    for r in repair_actions:
        q = r["example_id"]
        if q not in selected or (r["context_tokens"], r["omit_index"]) < (
                selected[q]["context_tokens"], selected[q]["omit_index"]):
            selected[q] = r
    assert len(selected) == 13
    actions.extend(("selected_repair", r) for r in selected.values())
    ids = {r["example_id"] for _, r in actions}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(
        LINEAGE / "sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl")
        if r["example_id"] in ids}
    cache = {}
    for path in sorted(P0.glob("shard_*_of_3/per_prefix.jsonl")):
        for row in read_jsonl(path):
            if row["example_id"] in ids:
                cache[row["example_id"], row["mask"]] = row
    assert len(orders) == len(atoms) == len(ids)
    details = []
    for group, row in actions:
        q = row["example_id"]
        order = orders[q]
        before, baseline = cache[q, mask(order, 8)], cache[q, mask(order, 9)]
        t = transition(before, baseline, row, atoms[q])
        if group == "safety":
            assert baseline["f1"] + EPS >= .90
            outcome = "break" if row["f1"] + EPS < .90 else "safe"
        else:
            assert baseline["f1"] + EPS < .90 and row["f1"] + EPS >= .90
            outcome = "selected_repair"
        details.append({"example_id": q, "group": group, "outcome": outcome,
                        "omit_index": row["omit_index"],
                        "candidate_index": row["candidate_index"],
                        "context_token_delta": row["context_tokens"] - row["baseline_tokens"],
                        **t})
    assert len(details) == 166
    summary = {"protocol": "SLOT-A0_CACHED_FUNCTIONAL_PROVENANCE",
               "safety_queries": len({r["example_id"] for r in details if r["group"] == "safety"}),
               "safety_actions": 153, "selected_stable_repair_queries": len(selected),
               "new_target_calls": 0, "new_training": False, "sealed_sets_read": False,
               "by_outcome": {}}
    for outcome in ("break", "safe", "selected_repair"):
        rows = [r for r in details if r["outcome"] == outcome]
        categories = Counter()
        for r in rows:
            if r["early_lost"] and r["late_lost"]:
                category = "mixed_early_and_late_loss"
            elif r["early_lost"]:
                category = "early_answer_loss_only"
            elif r["late_lost"]:
                category = "late_acquired_answer_loss_only"
            else:
                category = "no_correct_answer_loss"
            categories[category] += 1
        summary["by_outcome"][outcome] = {
            "actions": len(rows), "queries": len({r["example_id"] for r in rows}),
            "answer_loss_types": dict(categories),
            "any_new_correct": sum(r["new_correct"] > 0 for r in rows),
            "fewer_false_answers": sum(r["false_answer_delta"] < 0 for r in rows),
            "more_false_answers": sum(r["false_answer_delta"] > 0 for r in rows),
            "depth8_success_090": sum(r["depth8_f1"] + EPS >= .90 for r in rows),
        }
    summary["contract_notes"] = [
        "The depth-10 context is canonically re-rendered; there is no carried Target state, so a depth-10 answer difference cannot establish persistent reveal-order effects.",
        "Depth-9 swaps simultaneously defer default text and add candidate text. These answer transitions do not causally separate the two interventions.",
        "The 13 repair actions are outcome-selected oracle examples; the 153 safety actions cluster within 32 queries. Counts are descriptive, not independent-query estimates.",
        "Gold matching is retrospective and cannot be used as a deployment-visible feature.",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "per_action.jsonl").open("w") as handle:
        for r in details:
            handle.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
