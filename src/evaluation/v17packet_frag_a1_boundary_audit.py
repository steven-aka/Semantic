"""Lossless abbreviation-boundary audit of R1's existing action fragments."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path("results/v2_rank_then_cut")
OUT = ROOT / "v17packet_frag_a1_boundary_audit"
SPLIT = re.compile(r"(?<=[.!?])\s+")
INITIAL = re.compile(r"(?:^|\W)[A-Za-z]\.$")
WORD = re.compile(r"\b\w+\b")


def read(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def components(proof):
    parts = [x.strip() for x in SPLIT.split(proof) if x.strip()]
    assert " ".join(parts) == proof
    return parts


def merge_initial_boundaries(parts):
    units = []
    groups = []
    active = ""
    members = []
    for index, part in enumerate(parts):
        active = f"{active} {part}" if active else part
        members.append(index)
        if INITIAL.search(part) and index + 1 < len(parts):
            continue
        units.append(active)
        groups.append(members)
        active, members = "", []
    if active:
        units.append(active)
        groups.append(members)
    assert " ".join(units) == " ".join(parts)
    assert [i for group in groups for i in group] == list(range(len(parts)))
    return units, groups


def main():
    query_ids = {r["example_id"] for r in read(ROOT / "v17packet_r1_label_scale512/per_query.jsonl")}
    data = {r["example_id"]: r for r in read(
        ROOT / "v17sel_b2b_lineage_clean_holdout/data/v10_v12_train_train_clean.jsonl")
        if r["example_id"] in query_ids}
    order = {r["example_id"]: r["decoded_order"] for r in read(
        ROOT / "v17sel_b2b_lineage_clean_holdout/sel_c0_train_v8_rollouts.jsonl")
        if r["example_id"] in query_ids}
    assert len(query_ids) == len(data) == len(order) == 512
    stats = Counter()
    examples = []
    merged_members = {}
    for q in sorted(query_ids):
        packet = data[q]["packet_texts"][order[q][9]]
        title, sep, proof = packet.partition("\nSource evidence: ")
        assert title.startswith("Document title: ") and sep
        parts = components(proof)
        if len(parts) < 2:
            continue
        units, groups = merge_initial_boundaries(parts)
        merged_members[q] = {i for group in groups if len(group) > 1 for i in group}
        stats["splittable_queries"] += 1
        stats["old_candidates"] += len(parts)
        stats["new_units"] += len(units)
        stats["old_short"] += sum(len(WORD.findall(p)) <= 3 for p in parts)
        stats["new_short"] += sum(len(WORD.findall(p)) <= 3 for p in units)
        stats["queries_with_merges"] += len(units) < len(parts)
        stats["merged_boundaries"] += len(parts) - len(units)
        for unit, group in zip(units, groups):
            if len(group) > 1:
                examples.append({"example_id": q, "old_indices": group,
                                 "old_parts": [parts[i] for i in group], "merged_unit": unit})
    assert stats["splittable_queries"] == 424 and stats["old_candidates"] == 1485
    decisive = [r for r in read(ROOT / "v17packet_insert_c0_visible_signal/per_action.jsonl")
                if r["outcome"] in ("repair", "break")]
    impacted = [r for r in decisive if r["candidate_index"] in merged_members.get(r["example_id"], set())]
    assert len(decisive) == 21 and len(impacted) == 3
    summary = {"protocol": "FRAG-A1_LOSSLESS_ABBREVIATION_BOUNDARY_AUDIT",
               "new_target_calls": 0, "model_training": False, "sealed_sets_read": False,
               "counts": dict(stats),
               "decisive_pilot_actions_affected": len(impacted),
               "decisive_pilot_queries_affected": len({r["example_id"] for r in impacted}),
               "contract": "Merge only a boundary after a single-letter abbreviation; preserve original title separately and every proof character in source order. Bare section headings remain unresolved rather than automatically filtered.",
               "limitations": ["Corrected units are a proposed action space, not measured Target improvements.",
                               "One-letter abbreviation merging does not solve headings, initials separated without periods, decimal or all discourse-boundary errors."],
               "decision": "DEFER_BROAD_CLEANED_REPLAY_ONLY_ONE_DECISIVE_QUERY_AFFECTED"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "merged_examples.jsonl").open("w") as handle:
        for row in examples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
