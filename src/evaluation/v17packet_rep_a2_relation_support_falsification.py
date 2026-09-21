"""Small, explicitly post-hoc relation-support falsification on cached actions.

The hand-reviewed labels below concern whether the *inserted title and exact
sentence alone* assert every relation in the query. They are not independent
training labels, and they must never be used as deployment/gold features.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.evaluation.v17packet_r0_targeted_atomicity_pilot import segments


ROOT = Path("results/v2_rank_then_cut")
OUT = ROOT / "v17packet_rep_a2_relation_support_falsification"

# Manual, inspectable full-relation judgments. `partial` means only one of a
# conjunctive query's requirements is explicitly stated. Text is recovered
# from the frozen packet and emitted below for verification.
LABELS = {
    ("16159__wikidata_simple__train", 0): "none",
    ("59092__wikitables_composition__train", 6): "full",
    ("22398__wikidata_simple__train", 0): "none",
    ("22398__wikidata_simple__train", 1): "full",
    ("22398__wikidata_simple__train", 2): "none",
    ("27727__wikidata_simple__train", 0): "none",
    ("27727__wikidata_simple__train", 1): "none",
    ("27727__wikidata_simple__train", 2): "none",
    ("55423__wikidata_intersection__train", 0): "partial",
    ("22054__wikidata_simple__train", 0): "none",
    ("22054__wikidata_simple__train", 1): "none",
    ("53859__wikidata_intersection__train", 0): "none",
    ("53859__wikidata_intersection__train", 1): "none",
    ("53859__wikidata_intersection__train", 2): "partial",
    ("53859__wikidata_intersection__train", 4): "none",
    ("53859__wikidata_intersection__train", 5): "none",
    ("53909__wikidata_intersection__train", 0): "none",
    ("53909__wikidata_intersection__train", 2): "none",
    ("53909__wikidata_intersection__train", 4): "none",
    ("53909__wikidata_intersection__train", 5): "none",
    ("22814__wikidata_simple__train", 0): "none",
    # Outcome controls: clear support can occur without a 0.90 repair.
    ("22814__wikidata_simple__train", 1): "full",
    ("53909__wikidata_intersection__train", 1): "full",
    ("16267__wikidata_simple__train", 5): "full",
    ("86__wikidata_simple__train", 1): "full",
}


def read(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def main():
    actions = {(r["example_id"], r["candidate_index"]): r for r in read(
        ROOT / "v17packet_insert_c0_visible_signal/per_action.jsonl")}
    assert len(actions) == 84 and set(LABELS) <= set(actions)
    decisive = {key for key, row in actions.items() if row["outcome"] in ("repair", "break")}
    assert decisive <= set(LABELS) and len(decisive) == 21
    ids = {q for q, _ in LABELS}
    data = {r["example_id"]: r for r in read(
        ROOT / "v17sel_b2b_lineage_clean_holdout/data/v10_v12_train_train_clean.jsonl")
        if r["example_id"] in ids}
    order = {r["example_id"]: r["decoded_order"] for r in read(
        ROOT / "v17sel_b2b_lineage_clean_holdout/sel_c0_train_v8_rollouts.jsonl")
        if r["example_id"] in ids}
    assert len(data) == len(order) == len(ids)
    detail = []
    for (q, index), label in sorted(LABELS.items()):
        source = data[q]
        packet = source["packet_texts"][order[q][9]]
        sentence = segments(packet)[index]
        detail.append({"example_id": q, "candidate_index": index,
                       "question": source["question"], "inserted_text": sentence,
                       "relation_support": label,
                       "outcome": actions[q, index]["outcome"],
                       "delta_f1": actions[q, index]["delta_f1"]})
    by_outcome = defaultdict(Counter)
    for row in detail:
        by_outcome[row["outcome"]][row["relation_support"]] += 1
    assert sum(by_outcome["repair"].values()) == 16
    assert sum(by_outcome["break"].values()) == 5
    # The full R1 candidate pool is checked separately for sentence-split
    # fragments. This is a reproducible action-construction diagnostic, not
    # a semantic-support judgment.
    r1 = {r["example_id"]: r for r in read(ROOT / "v17packet_r1_label_scale512/per_query.jsonl")}
    all_ids = set(r1)
    full_data = {r["example_id"]: r for r in read(
        ROOT / "v17sel_b2b_lineage_clean_holdout/data/v10_v12_train_train_clean.jsonl")
        if r["example_id"] in all_ids}
    full_order = {r["example_id"]: r["decoded_order"] for r in read(
        ROOT / "v17sel_b2b_lineage_clean_holdout/sel_c0_train_v8_rollouts.jsonl")
        if r["example_id"] in all_ids}
    assert len(full_data) == len(full_order) == len(r1) == 512
    fragment = Counter()
    for row in read(ROOT / "v17packet_r1_label_scale512/per_sentence.jsonl"):
        q, index = row["example_id"], row["sentence_index"]
        sentence = segments(full_data[q]["packet_texts"][full_order[q][9]])[index]
        proof = sentence.partition("\nSource evidence: ")[2]
        short = len(re.findall(r"\b\w+\b", proof)) <= 3
        before = r1[q]["baseline"]["hits"][3]
        after = row["f1"] + 1e-6 >= .90
        outcome = "repair" if not before and after else "break" if before and not after else "other"
        fragment["total"] += 1
        fragment[f"{outcome}_total"] += 1
        fragment["short_total"] += short
        fragment[f"{outcome}_short"] += short
    assert fragment["total"] == 1485
    summary = {"protocol": "REP-A2_POSTHOC_RELATION_SUPPORT_FALSIFICATION",
               "scope": "all 21 decisive SLOT-B0 actions plus four clear-support controls",
               "new_target_calls": 0, "model_training": False, "sealed_sets_read": False,
               "relation_support_by_outcome": {k: dict(v) for k, v in by_outcome.items()},
               "full_support_repair_recall_on_annotated_actions": by_outcome["repair"]["full"] / 16,
               "r1_short_fragment_diagnostic": dict(fragment),
               "judgment_contract": "Exact inserted title+sentence explicitly states every query relation for the titled answer; partial states only some; no inference from gold/proof block.",
               "limitations": [
                   "Labels were inspected after outcomes were known; this is a necessary-condition check, not an unbiased predictive validation.",
                   "Only decisive actions and four controls are hand-reviewed; other 59 actions have no relation-support labels.",
                   "A single sentence may still change the Target through priming, entity cues, ordering, or interaction with existing prefix text.",
                   "Manual judgments need independent blinded review before use as a training set.",
               ],
               "decision": "STOP_AUTOMATIC_PROOF_TO_SENTENCE_SUPPORT_LABELING; do not train support encoder from current block-level certificates"}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_action.jsonl").open("w", encoding="utf-8") as handle:
        for row in detail:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
