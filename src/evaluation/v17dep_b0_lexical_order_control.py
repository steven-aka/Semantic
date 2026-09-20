"""Frozen, train-only raw-evidence lexical trajectory control."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.build_v17sel_c0_train_candidates import exact_values, outcome
from src.data.schemas import read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata


TOKEN = re.compile(r"[a-z0-9]+")


def lexical_order(question: str, packet_texts: list[str], v8_order: list[int]) -> list[int]:
    if len(packet_texts) != 12 or sorted(v8_order) != list(range(12)):
        raise ValueError("expected twelve lossless packets and valid V8 order")
    terms = set(TOKEN.findall(question.lower()))
    documents = [Counter(TOKEN.findall(text.lower())) for text in packet_texts]
    lengths = [sum(document.values()) for document in documents]
    average_length = sum(lengths) / 12
    document_frequency = Counter(term for document in documents for term in document)
    scores = []
    for document, length in zip(documents, lengths):
        score = 0.0
        for term in terms:
            frequency = document[term]
            if frequency:
                inverse_frequency = math.log(1 + (12 - document_frequency[term] + .5) /
                                             (document_frequency[term] + .5))
                score += inverse_frequency * frequency * 2.2 / (frequency + 1.2 * (.25 + .75 * length / average_length))
        scores.append(score)
    v8_position = {packet: position for position, packet in enumerate(v8_order)}
    return sorted(range(12), key=lambda packet: (-scores[packet], v8_position[packet], packet))


def prefix_outcome(order: list[int], fidelity: list[float], tokens: list[int], attainable: set[float]) -> dict:
    return outcome(order, fidelity, tokens, attainable)


def aggregate(rows: list[dict], key: str) -> dict:
    values = [row[key] for row in rows]
    successful = [value for value in values if value["success"][3]]
    complete = [value for value in values if value["complete"]]
    return {
        "queries": len(rows),
        "oracle_090_success": len(successful),
        "oracle_complete": len(complete),
        "mean_earliest_090_token_fraction_if_success": mean(value["earliest_tokens"][3] / row["full_tokens"] for row, value in zip(rows, values) if value["success"][3]),
        "mean_failure_penalized_090_token_fraction": mean((value["earliest_tokens"][3] / row["full_tokens"]) if value["success"][3] else 1.0 for row, value in zip(rows, values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_TRAIN_ONLY_READ_ONLY":
        raise ValueError("unfrozen control")
    source = {row["example_id"]: row for row in read_jsonl(args.source)}
    candidates = [row for row in read_jsonl(args.candidates) if fold(row["example_id"]) != 4]
    rollouts = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    if len(candidates) != 1421:
        raise ValueError("unexpected train-only population")
    table = []
    for count, row in enumerate(candidates, 1):
        query = row["example_id"]
        item = source[query]
        baseline = rollouts[query]
        lexical = lexical_order(item["question"], item["packet_texts"], baseline)
        fidelity, tokens = exact_values(Path(args.exact_dir) / f"{query}.jsonl")
        levels = set(row["attainable_levels"])
        v8 = prefix_outcome(baseline, fidelity, tokens, levels)
        if v8 != {key: row["candidates"][0][key] for key in v8}:
            raise ValueError(f"V8 cached outcome mismatch: {query}")
        lexical_result = prefix_outcome(lexical, fidelity, tokens, levels)
        rank1 = {key: row["candidates"][1][key] for key in v8}
        table.append({"example_id": query, "full_tokens": row["full_tokens"],
                      "v8_order": baseline, "lexical_order": lexical,
                      "v8": v8, "lexical": lexical_result, "v13_rank1": rank1})
        if count % 500 == 0:
            print(json.dumps({"loaded": count, "total": len(candidates)}), flush=True)
    paired = [row for row in table if row["v8"]["success"][3] and row["lexical"]["success"][3]]
    both_complete = [row for row in table if row["v8"]["complete"] and row["lexical"]["complete"]]
    summary = {
        "protocol": config["protocol"], "role": "train-only exact-oracle trajectory control, not deployed cutoff",
        "arms": {name: aggregate(table, name) for name in ("v8", "lexical", "v13_rank1")},
        "lexical_vs_v8": {
            "repair_090": sum(not row["v8"]["success"][3] and row["lexical"]["success"][3] for row in table),
            "break_090": sum(row["v8"]["success"][3] and not row["lexical"]["success"][3] for row in table),
            "complete_gain": sum(not row["v8"]["complete"] and row["lexical"]["complete"] for row in table),
            "complete_break": sum(row["v8"]["complete"] and not row["lexical"]["complete"] for row in table),
            "paired_090_queries": len(paired),
            "mean_paired_earliest_090_token_delta": mean(row["lexical"]["earliest_tokens"][3] - row["v8"]["earliest_tokens"][3] for row in paired),
            "paired_complete_queries": len(both_complete),
            "mean_paired_oracle_cumulative_token_delta": mean(row["lexical"]["oracle_ordered_complete_cumulative_tokens"] - row["v8"]["oracle_ordered_complete_cumulative_tokens"] for row in both_complete),
        },
        "fold4_611_read": False, "holdout_581_read": False, "internal300_read": False,
        "development_read": False, "confirmation_read": False, "new_target_calls": 0,
        "artifacts": {"config_sha256": sha256(args.config), "source_sha256": sha256(args.source),
                      "candidates_sha256": sha256(args.candidates), "rollouts_sha256": sha256(args.rollouts)},
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_query.jsonl", table)
    write_metadata(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
