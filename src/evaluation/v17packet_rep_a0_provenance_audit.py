"""Zero-Target-call audit of certified proof provenance on R1 and SLOT-B0.

Gold annotations are used only for a retrospective ceiling, never as a
deployment feature or sentence-level entailment label.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("results/v2_rank_then_cut")
LINEAGE = ROOT / "v17sel_b2b_lineage_clean_holdout"
OUT = ROOT / "v17packet_rep_a0_provenance_audit"


def read(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def split_proof(packet: str):
    title, separator, proof = packet.partition("\nSource evidence: ")
    if not separator or not title.startswith("Document title: "):
        raise ValueError("unexpected packet format")
    return title.removeprefix("Document title: "), proof


def sentences(proof: str):
    parts = [x.strip() for x in re.split(r"(?<=[.!?])\s+", proof) if x.strip()]
    assert " ".join(parts) == proof
    return parts


def main():
    r1 = list(read(ROOT / "v17packet_r1_label_scale512/per_query.jsonl"))
    actions = list(read(ROOT / "v17packet_insert_c0_visible_signal/per_action.jsonl"))
    ids = {x["example_id"] for x in r1}
    data = {x["example_id"]: x for x in read(LINEAGE / "data/v10_v12_train_train_clean.jsonl") if x["example_id"] in ids}
    order = {x["example_id"]: x["decoded_order"] for x in read(LINEAGE / "sel_c0_train_v8_rollouts.jsonl") if x["example_id"] in ids}
    atoms = {x["example_id"]: x["answer_atoms"] for x in read(Path("data/units/qampari_rank_v2_candidates5000_annotations.jsonl")) if x["example_id"] in ids}
    assert len(data) == len(order) == len(atoms) == len(r1) == 512

    lookup = {}
    for q in ids:
        packet = data[q]["packet_texts"][order[q][9]]
        title, proof = split_proof(packet)
        matches = [i for i, atom in enumerate(atoms[q]) if normalize(atom["answer_text"]) == normalize(title)]
        exact = len(matches) == 1 and normalize(atoms[q][matches[0]]["proof"]) == normalize(proof)
        terms = atoms[q][matches[0]]["certificate"].get("relation_term_overlap", []) if matches else []
        lookup[q] = {"title": title, "proof": proof, "gold_atom_indices": matches,
                     "exact_proof_match": exact, "sentences": sentences(proof),
                     "certificate_terms": terms}

    by_query = defaultdict(list)
    relation_hits = Counter()
    output = []
    for row in actions:
        q, i = row["example_id"], row["candidate_index"]
        source = lookup[q]
        sentence = source["sentences"][i]
        # Certificate term overlap is a heuristic provenance diagnostic,
        # not an entailment label or a deployable feature.
        hit = any(re.search(r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)", sentence.casefold())
                  for term in source["certificate_terms"] if term)
        item = {"example_id": q, "candidate_index": i, "outcome": row["outcome"],
                "delta_f1": row["delta_f1"], "gold_title_match": bool(source["gold_atom_indices"]),
                "exact_proof_match": source["exact_proof_match"],
                "certificate_term_in_sentence": hit}
        by_query[q].append(item)
        output.append(item)
        relation_hits[row["outcome"], hit] += 1

    failures = [x for x in r1 if not x["baseline"]["hits"][3]]
    repairable = [x for x in failures if x["quality_oracle"]["hits"][3]]
    unrepairable = [x for x in failures if not x["quality_oracle"]["hits"][3]]
    strict_pairs = [(a, b) for rows in by_query.values() for a in rows for b in rows
                    if a["outcome"] == "repair" and b["outcome"] != "repair"]
    summary = {
        "scope": "design-exposed train512 R1 and selected train24 SLOT-B0 only",
        "target_calls": 0,
        "r1_queries": len(r1),
        "r1_rank10_gold_title": sum(bool(lookup[x["example_id"]]["gold_atom_indices"]) for x in r1),
        "r1_rank10_exact_certified_proof": sum(lookup[x["example_id"]]["exact_proof_match"] for x in r1),
        "r1_failures": len(failures),
        "r1_repairable_failures": len(repairable),
        "r1_repairable_gold_title": sum(bool(lookup[x["example_id"]]["gold_atom_indices"]) for x in repairable),
        "r1_unrepairable_failures": len(unrepairable),
        "r1_unrepairable_gold_title": sum(bool(lookup[x["example_id"]]["gold_atom_indices"]) for x in unrepairable),
        "slot_b0_queries": len(by_query), "slot_b0_actions": len(actions),
        "slot_b0_gold_title_actions": sum(x["gold_title_match"] for x in output),
        "slot_b0_exact_certified_proof_actions": sum(x["exact_proof_match"] for x in output),
        "slot_b0_queries_with_variable_gold_novelty": sum(len({x["gold_title_match"] for x in rows}) > 1 for rows in by_query.values()),
        "slot_b0_strict_repair_nonrepair_pairs": len(strict_pairs),
        "slot_b0_gold_novelty_discriminated_pairs": sum(a["gold_title_match"] != b["gold_title_match"] for a, b in strict_pairs),
        "slot_b0_certificate_term_contingency": {f"{outcome}|{hit}": n for (outcome, hit), n in sorted(relation_hits.items())},
        "limitations": [
            "Certified proof provenance is at answer-atom/block level; a split sentence is not automatically an entailing support sentence.",
            "Document title reveals a gold answer in almost every rank10 packet; this is a retrospective upper-bound feature, not a permissible gold input.",
            "The 24-query pilot was baseline-balanced and design-exposed; action counts are not independent query counts.",
            "Certificate relation-term overlap is a construction heuristic, not logical entailment.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "per_action.jsonl").open("w", encoding="utf-8") as handle:
        for item in output:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
