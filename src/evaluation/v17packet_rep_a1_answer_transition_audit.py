"""Decompose cached protected-insertion effects into gold-answer transitions.

This is a retrospective train-side mechanism audit, not a deployable policy.
It makes no target calls and reads no sealed evaluation set.
"""
from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.evaluation.qampari_metrics import (normalize_list_answer, parse_cached_list_prediction,
                                            qampari_list_metrics)


ROOT = Path("results/v2_rank_then_cut")
OUT = ROOT / "v17packet_rep_a1_answer_transition_audit"


def read(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def matched_atoms(prediction, atoms):
    alias_to_atom = {}
    for i, atom in enumerate(atoms):
        for alias in [atom["answer_text"], *atom.get("aliases", [])]:
            alias_to_atom.setdefault(normalize_list_answer(str(alias)), i)
    predictions = {normalize_list_answer(x) for x in parse_cached_list_prediction(prediction)}
    return {alias_to_atom[x] for x in predictions if x in alias_to_atom}, predictions


def main():
    actions = list(read(ROOT / "v17packet_insert_c0_visible_signal/per_action.jsonl"))
    ids = {x["example_id"] for x in actions}
    target = {(x["example_id"], x["candidate_index"]): x for x in read(ROOT / "v17packet_slot_b0_pilot/per_call.jsonl")}
    atoms = {x["example_id"]: x["answer_atoms"] for x in read("data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if x["example_id"] in ids}
    data = {x["example_id"]: x for x in read(ROOT / "v17sel_b2b_lineage_clean_holdout/data/v10_v12_train_train_clean.jsonl") if x["example_id"] in ids}
    order = {x["example_id"]: x["decoded_order"] for x in read(ROOT / "v17sel_b2b_lineage_clean_holdout/sel_c0_train_v8_rollouts.jsonl") if x["example_id"] in ids}
    baseline = {}
    for path in glob.glob(str(ROOT / "v17canon_p0_fresh_v8_prefix_chain/shard_*_of_3/per_prefix.jsonl")):
        for row in read(path):
            if row["example_id"] in ids and row["depth"] == 9:
                assert row["example_id"] not in baseline
                baseline[row["example_id"]] = row
    assert len(ids) == len(atoms) == len(data) == len(order) == len(baseline) == 24
    assert len(actions) == len(target) == 84

    detail = []
    for row in actions:
        q, index = row["example_id"], row["candidate_index"]
        assert abs(qampari_list_metrics(parse_cached_list_prediction(baseline[q]["prediction"]), atoms[q])["f1"] - baseline[q]["f1"]) < 1e-8
        assert abs(qampari_list_metrics(parse_cached_list_prediction(target[q, index]["prediction"]), atoms[q])["f1"] - target[q, index]["f1"]) < 1e-8
        before, before_text = matched_atoms(baseline[q]["prediction"], atoms[q])
        after, after_text = matched_atoms(target[q, index]["prediction"], atoms[q])
        title = data[q]["packet_texts"][order[q][9]].split("\n", 1)[0].removeprefix("Document title: ")
        title_matches = [i for i, atom in enumerate(atoms[q]) if atom["answer_text"].casefold() == title.casefold()]
        assert len(title_matches) == 1
        title_index = title_matches[0]
        added, lost = after - before, before - after
        item = {"example_id": q, "candidate_index": index, "outcome": row["outcome"],
                "delta_f1": row["delta_f1"], "candidate_atom_index": title_index,
                "candidate_atom_added": title_index in added,
                "other_gold_atoms_added": len(added - {title_index}),
                "gold_atoms_lost": len(lost),
                "gold_atoms_before": len(before), "gold_atoms_after": len(after),
                "prediction_count_before": len(before_text), "prediction_count_after": len(after_text),
                "added_atom_indices": sorted(added), "lost_atom_indices": sorted(lost)}
        detail.append(item)

    counts = defaultdict(Counter)
    query_sets = defaultdict(lambda: defaultdict(set))
    for item in detail:
        outcome = item["outcome"]
        c = counts[outcome]
        c["actions"] += 1
        c["candidate_atom_added_actions"] += item["candidate_atom_added"]
        c["other_gold_added_actions"] += item["other_gold_atoms_added"] > 0
        c["other_gold_added_total"] += item["other_gold_atoms_added"]
        c["gold_lost_actions"] += item["gold_atoms_lost"] > 0
        c["gold_lost_total"] += item["gold_atoms_lost"]
        query_sets[outcome]["all"].add(item["example_id"])
        if item["candidate_atom_added"]:
            query_sets[outcome]["candidate_atom_added"].add(item["example_id"])
        if item["other_gold_atoms_added"]:
            query_sets[outcome]["other_gold_added"].add(item["example_id"])
        if item["gold_atoms_lost"]:
            query_sets[outcome]["gold_lost"].add(item["example_id"])

    summary = {"scope": "design-exposed, baseline-balanced SLOT-B0 train24",
               "target_calls": 0, "actions": len(detail), "queries": len(ids),
               "by_outcome": {outcome: {**dict(counts[outcome]),
                   "queries": len(query_sets[outcome]["all"]),
                   "queries_candidate_atom_added": len(query_sets[outcome]["candidate_atom_added"]),
                   "queries_other_gold_added": len(query_sets[outcome]["other_gold_added"]),
                   "queries_gold_lost": len(query_sets[outcome]["gold_lost"])}
                   for outcome in sorted(counts)},
               "limitations": [
                   "Repeated actions from a query are correlated; action rates do not estimate population probabilities.",
                   "Gold-answer matching is retrospective and cannot be used as a deployment feature.",
                   "Answer transitions reveal outcome mechanisms but do not identify whether evidence semantics, order, or decoding caused them.",
                   "Title/proof provenance is block-level; sentence-level relation support remains unaudited.",
               ]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (OUT / "per_action.jsonl").open("w", encoding="utf-8") as handle:
        for item in detail:
            handle.write(json.dumps(item) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
