"""Cached-answer mechanism audit for the frozen R1 sentence substitution."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import normalize_list_answer, qampari_list_metrics
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render, segments


ROOT = Path("results/v2_rank_then_cut")
LINEAGE = ROOT / "v17sel_b2b_lineage_clean_holdout"
R1 = ROOT / "v17packet_r1_label_scale512"
P0 = ROOT / "v17canon_p0_fresh_v8_prefix_chain"
OUT = ROOT / "v17packet_mech_a0_cached_audit"
LEVEL = 0.90
EPSILON = 1e-6


def stored_answers(prediction: str) -> list[str]:
    # Cache predictions are already parsed and joined with " # ". Do not
    # invoke parse_answer a second time: e.g. "No. 13 Squadron" would be
    # incorrectly cut to "No." by the raw-generation parser.
    return [part.strip() for part in prediction.split(" # ") if part.strip()]


def normalized_answers(prediction: str) -> set[str]:
    return {normalize_list_answer(x) for x in stored_answers(prediction)}


def matched_atoms(prediction: str, atoms: list[dict]) -> set[int]:
    aliases = {}
    for index, atom in enumerate(atoms):
        for text in [atom["answer_text"], *atom.get("aliases", [])]:
            aliases.setdefault(normalize_list_answer(str(text)), index)
    return {aliases[x] for x in normalized_answers(prediction) if x in aliases}


def gold_mentions(text: str, atoms: list[dict]) -> set[int]:
    # Lexical heuristic only: an answer name may be mentioned without
    # satisfying the query relation, or not mentioned despite valid support.
    haystack = normalize_list_answer(text)
    found = set()
    for i, atom in enumerate(atoms):
        for alias in [atom["answer_text"], *atom.get("aliases", [])]:
            needle = normalize_list_answer(str(alias))
            if len(needle) >= 4 and re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack):
                found.add(i)
                break
    return found


def main() -> None:
    rows = list(read_jsonl(R1 / "per_query.jsonl"))
    ids = {r["example_id"] for r in rows}
    manifest = {r["example_id"]: r for r in json.loads((R1 / "manifest.json").read_text())["cases"]}
    calls = {(r["example_id"], r["sentence_index"]): r for r in read_jsonl(R1 / "per_sentence.jsonl")}
    data = {r["example_id"]: r for r in read_jsonl(LINEAGE / "data/v10_v12_train_train_clean.jsonl")
            if r["example_id"] in ids}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(
        LINEAGE / "sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    cache = {}
    for path in sorted(P0.glob("shard_*_of_3/per_prefix.jsonl")):
        for item in read_jsonl(path):
            q = item["example_id"]
            if q in ids:
                cache[q, item["mask"]] = item
    assert len(rows) == len(manifest) == len(data) == len(orders) == len(atoms) == 512
    details = []
    for row in rows:
        q = row["example_id"]
        order = orders[q]
        packets = data[q]["packet_texts"]
        m8, m9 = mask(order, 8), mask(order, 9)
        b, d = cache[q,m9], cache[q,m8]
        assert (b["f1"] + EPSILON >= LEVEL) == row["baseline"]["hits"][3]
        assert abs(qampari_list_metrics(stored_answers(b["prediction"]), atoms[q])["f1"] - b["f1"]) < 1e-8
        fragments = segments(packets[order[9]])
        assert len(fragments) == len(row["sentence_options"])
        eligible = [a for a in row["sentence_options"]
                    if a["context_tokens"] <= row["baseline"]["context_tokens"]]
        repairs = [a for a in eligible if not row["baseline"]["hits"][3] and a["hits"][3]]
        kind = ("already_good" if row["baseline"]["hits"][3] else
                "repairable_failure" if repairs else "unrepairable_failure")
        baseline_text = render(packets, m9)
        baseline_mentions = gold_mentions(baseline_text, atoms[q])
        option_mentions = {a["index"]: gold_mentions(fragments[a["index"]], atoms[q])
                           for a in eligible}
        detail = {"example_id": q, "kind": kind, "baseline_f1": b["f1"],
                  "depth8_f1": d["f1"], "depth8_success_090": d["f1"]+EPSILON >= LEVEL,
                  "baseline_correct_count": len(matched_atoms(b["prediction"], atoms[q])),
                  "baseline_predicted_count": len(normalized_answers(b["prediction"])),
                  "candidate_count": len(eligible), "repair_action_count": len(repairs),
                  "baseline_gold_mentions": len(baseline_mentions),
                  "any_candidate_novel_gold_mention": any(bool(v-baseline_mentions)
                                                           for v in option_mentions.values())}
        if repairs:
            chosen = next((a for a in repairs if a["index"] == row["strict_index"]), None)
            assert chosen is not None
            ad = calls[q,chosen["index"]]
            assert abs(ad["f1"] - qampari_list_metrics(
                stored_answers(ad["prediction"]), atoms[q])["f1"]) < 1e-8
            bpred, adpred = normalized_answers(b["prediction"]), normalized_answers(ad["prediction"])
            bcorrect, adcorrect = matched_atoms(b["prediction"], atoms[q]), matched_atoms(ad["prediction"], atoms[q])
            bfalse = len(bpred) - len(bcorrect)
            adfalse = len(adpred) - len(adcorrect)
            new_correct, removed_correct = adcorrect-bcorrect, bcorrect-adcorrect
            detail.update({"selected_sentence_index": chosen["index"],
                           "repair_f1": ad["f1"], "f1_gain": ad["f1"]-b["f1"],
                           "baseline_f1_margin": b["f1"]-LEVEL,
                           "repair_f1_margin": ad["f1"]-LEVEL,
                           "new_correct_atom_count": len(new_correct),
                           "removed_correct_atom_count": len(removed_correct),
                           "false_answer_count_delta": adfalse-bfalse,
                           "new_gold_mentions_in_sentence": len(option_mentions[chosen["index"]]-baseline_mentions),
                           "new_correct_atoms_mentioned_in_sentence": len(new_correct & option_mentions[chosen["index"]]),
                           "delay_only_success": d["f1"]+EPSILON >= LEVEL})
        details.append(detail)
    repair = [r for r in details if r["kind"] == "repairable_failure"]
    no_repair = [r for r in details if r["kind"] == "unrepairable_failure"]
    assert len(repair) == 39 and len(no_repair) == 92
    mechanism = Counter()
    for r in repair:
        added = r["new_correct_atom_count"] > 0
        removed_false = r["false_answer_count_delta"] < 0
        mechanism[("new_correct" if added else "no_new_correct") +
                  ("_and_fewer_false" if removed_false else "_without_fewer_false")] += 1
    summary = {"protocol": "V17-PACKET-MECH-A0_CACHED_ANSWER_AND_EVIDENCE_AUDIT",
               "queries": len(rows), "repairable_failures": len(repair),
               "unrepairable_failures": len(no_repair),
               "repair_answer_change_types": dict(mechanism),
               "repair_depth8_already_successful": sum(r["delay_only_success"] for r in repair),
               "repair_sentence_new_gold_mention": sum(r["new_gold_mentions_in_sentence"]>0 for r in repair),
               "repair_new_correct_atom_mentioned_in_sentence": sum(r["new_correct_atoms_mentioned_in_sentence"]>0 for r in repair),
               "unrepairable_any_candidate_novel_gold_mention": sum(r["any_candidate_novel_gold_mention"] for r in no_repair),
               "repairable_any_candidate_novel_gold_mention": sum(r["any_candidate_novel_gold_mention"] for r in repair),
               "new_target_calls": 0, "new_training": False, "sealed_sets_read": False,
               "limitations": ["Gold lexical mentions are retrospective diagnostics, not evidence of a supported query relation or deployable features.",
                               "R1 repair actions were selected using single-run Target outcomes; paired replay is needed before causal claims.",
                               "D uses the cached V8 depth-8 state and changes context length as well as evidence."]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "per_query.jsonl").open("w") as handle:
        for r in details:
            handle.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
