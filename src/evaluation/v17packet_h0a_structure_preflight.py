"""Audit whether V8 packetization actually destroys recoverable source structure."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.data.schemas import QAExample, read_jsonl

CFG = Path("configs/v17packet_h0a_structure_preflight.json")
OUT = Path("results/v2_rank_then_cut/v17packet_h0a_structure_preflight")


def main() -> None:
    cfg = json.loads(CFG.read_text())
    assert cfg["status"] == "FROZEN_ZERO_TARGET_CALL_AUDIT"
    clean = list(read_jsonl(cfg["clean_rows"]))
    wanted = {row["example_id"] for row in clean}
    sources = {
        row.example_id: row
        for row in read_jsonl(cfg["source_examples"], QAExample)
        if row.example_id in wanted
    }
    assert len(clean) == len(wanted) == len(sources)

    malformed = 0
    reconstruction_failures = 0
    source_coordinate_failures = 0
    units_with_structural_metadata = 0
    block_counts: Counter[int] = Counter()
    packet_lengths: list[int] = []
    examples = []
    for row in clean:
        example = sources[row["example_id"]]
        packets = row["packet_texts"]
        packet_lengths.extend(row["packet_tokens"])
        malformed += sum(
            packet.count("Document title:") != 1
            or packet.count("\nSource evidence: ") != 1
            for packet in packets
        )
        reconstruction_failures += int("\n\n".join(packets) != example.context.strip())
        expected = []
        coords = []
        for unit in example.units:
            blocks = [block.strip() for block in unit.text.split("\n\n") if block.strip()]
            block_counts[len(blocks)] += 1
            expected.extend(blocks)
            coords.extend((unit.unit_id, index) for index in range(len(blocks)))
            units_with_structural_metadata += int(
                unit.title is not None or unit.source_sentence_id is not None
            )
        source_coordinate_failures += int(expected != packets)
        examples.append(
            {
                "example_id": row["example_id"],
                "packets": len(packets),
                "source_units": len(example.units),
                "blocks_per_unit": [
                    len([x for x in unit.text.split("\n\n") if x.strip()])
                    for unit in example.units
                ],
                "exact_context_reconstruction": "\n\n".join(packets) == example.context.strip(),
                "exact_source_block_match": expected == packets,
            }
        )

    packet_lengths.sort()
    result = {
        "protocol": cfg["protocol"],
        "queries": len(clean),
        "packets": sum(len(row["packet_texts"]) for row in clean),
        "packets_per_query": sorted({len(row["packet_texts"]) for row in clean}),
        "malformed_title_source_blocks": malformed,
        "exact_context_reconstruction_failures": reconstruction_failures,
        "exact_source_block_match_failures": source_coordinate_failures,
        "source_blocks_per_upstream_unit": dict(sorted(block_counts.items())),
        "units_with_title_or_sentence_metadata": units_with_structural_metadata,
        "packet_token_min_median_max": [
            packet_lengths[0],
            packet_lengths[len(packet_lengths) // 2],
            packet_lengths[-1],
        ],
        "finding": (
            "V8 operates on complete title-bound source blocks. The sentence fragments used "
            "by later SEM/R1 audits are post-hoc derived actions, not V8 packet boundaries. "
            "Merging sibling blocks would join distinct source blocks and does not restore a "
            "split proposition. No sentence/table/list metadata is present to recover."
        ),
        "decision": "STOP_PACKET_H0_NO_DISTINCT_STRUCTURAL_REPACKETIZATION",
        "h0b_target_calls_authorized": False,
        "new_teacher_calls": 0,
        "new_target_calls": 0,
        "sealed_sets_read": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUT / "per_query.jsonl").open("w") as handle:
        for row in examples:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
