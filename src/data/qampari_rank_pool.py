from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.data.qampari_ceiling_v2 import certified_atom
from src.data.schemas import QAExample, SemanticUnit, write_jsonl
from src.evaluation.qampari_metrics import normalize_list_answer
from src.representation.token_counter import count_tokens, load_tokenizer
from src.reproducibility import sha256, write_metadata


def _render(atom: Mapping[str, Any]) -> str:
    # Raw train proofs occasionally contain blank paragraphs.  Normalize only
    # whitespace so the full proof content remains one lossless atomic block.
    title = " ".join(str(atom["answer_text"]).split())
    proof = " ".join(str(atom["proof"]).split())
    return f"Document title: {title}\nSource evidence: {proof}"


def certified_records(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    seed: int,
    max_unit_tokens: int = 512,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records = []
    rejected = {
        "fewer_than_ten_certified_atoms": 0,
        "unit_too_long": 0,
    }
    for row in rows:
        atoms = []
        seen = set()
        for answer in row.get("answer_list", []):
            atom = certified_atom(row, answer, tokenizer)
            if atom is None:
                continue
            key = normalize_list_answer(str(atom["answer_text"]))
            if key in seen:
                continue
            atoms.append(atom)
            seen.add(key)
            if len(atoms) == 10:
                break
        if len(atoms) != 10:
            rejected["fewer_than_ten_certified_atoms"] += 1
            continue
        groups = [atoms[index:index + 2] for index in range(0, 10, 2)]
        rendered = ["\n\n".join(_render(atom) for atom in group) for group in groups]
        if any(count_tokens(tokenizer, text) > max_unit_tokens for text in rendered):
            rejected["unit_too_long"] += 1
            continue
        records.append(
            {
                "qid": str(row["qid"]),
                "question": str(row["question_text"]).strip(),
                "atoms": atoms,
                "relevant_groups": rendered,
                "sort_key": hashlib.sha256(f"{seed}:{row['qid']}".encode()).hexdigest(),
            }
        )
    records.sort(key=lambda value: (value["sort_key"], value["qid"]))
    return records, rejected


def build_rank_pool(
    records: list[dict[str, Any]],
    tokenizer: Any,
    *,
    n_examples: int,
    max_unit_tokens: int = 512,
) -> tuple[list[QAExample], list[dict[str, Any]]]:
    if n_examples > len(records) or n_examples <= 0:
        raise ValueError("requested rank pool is outside certified record count")
    examples = []
    annotations = []
    offset = max(1, len(records) // 2)
    for index, record in enumerate(records[:n_examples]):
        current_answers = {
            normalize_list_answer(str(atom["answer_text"])) for atom in record["atoms"]
        }
        distractor_record = None
        for delta in range(len(records)):
            candidate = records[(index + offset + delta) % len(records)]
            candidate_answers = {
                normalize_list_answer(str(atom["answer_text"]))
                for atom in candidate["atoms"][:2]
            }
            if candidate["qid"] != record["qid"] and not current_answers & candidate_answers:
                distractor_record = candidate
                break
        if distractor_record is None:
            raise RuntimeError(f"{record['qid']}: no disjoint deterministic distractor")
        distractor = "\n\n".join(_render(atom) for atom in distractor_record["atoms"][:2])
        if count_tokens(tokenizer, distractor) > max_unit_tokens:
            raise RuntimeError("certified distractor exceeds unit token limit")
        unit_rows = [
            {"text": text, "answer_indices": [2 * group, 2 * group + 1]}
            for group, text in enumerate(record["relevant_groups"])
        ] + [{"text": distractor, "answer_indices": []}]
        # Hash ordering avoids the source-file question-type blocks without
        # using labels or target outputs.
        unit_rows.sort(
            key=lambda unit: hashlib.sha256(
                f"{record['sort_key']}:{unit['text']}".encode()
            ).hexdigest()
        )
        units = [
            SemanticUnit(unit_id=unit_id, text=unit["text"], supporting=False)
            for unit_id, unit in enumerate(unit_rows)
        ]
        examples.append(
            QAExample(
                example_id=record["qid"],
                dataset="qampari_rank_then_cut_train_source",
                question=record["question"],
                answer=" # ".join(str(atom["answer_text"]) for atom in record["atoms"]),
                context="\n\n".join(unit.text for unit in units),
                units=units,
            )
        )
        annotations.append(
            {
                "example_id": record["qid"],
                "answer_atoms": record["atoms"],
                "unit_answer_indices": [unit["answer_indices"] for unit in unit_rows],
                "n_answer_atoms": 10,
                "n_relevant_units": 5,
                "n_distractor_units": 1,
                "evidence_certificate_pass": True,
            }
        )
    return examples, annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Build target-blind QAMPARI Rank-then-Cut pool")
    parser.add_argument("--input", required=True)
    parser.add_argument("--examples-output", required=True)
    parser.add_argument("--annotations-output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--tokenizer", default="models/Qwen3-8B")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--n-examples", type=int, default=5000)
    parser.add_argument("--max-unit-tokens", type=int, default=512)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    with Path(args.input).open(encoding="utf-8") as handle:
        records, rejected = certified_records(
            (json.loads(line) for line in handle if line.strip()),
            tokenizer,
            seed=args.seed,
            max_unit_tokens=args.max_unit_tokens,
        )
    examples, annotations = build_rank_pool(
        records,
        tokenizer,
        n_examples=args.n_examples,
        max_unit_tokens=args.max_unit_tokens,
    )
    write_jsonl(args.examples_output, examples)
    write_jsonl(args.annotations_output, annotations)
    write_metadata(
        args.metadata_output,
        {
            "stage": "v2_rank_then_cut_candidate_pool_before_target_inference",
            "source_split": "official_qampari_train",
            "input_sha256": sha256(args.input),
            "seed": args.seed,
            "strict_certified_records": len(records),
            "selected_candidates": len(examples),
            "rejection_counts": rejected,
            "target_outputs_observed_before_candidate_freeze": False,
            "selection": "sha256(seed:qid), then first n_examples",
            "distractor": "disjoint deterministic half-pool offset",
            "examples_sha256": sha256(args.examples_output),
            "annotations_sha256": sha256(args.annotations_output),
        },
    )
    print(json.dumps({"certified": len(records), "selected": len(examples), "rejected": rejected}))


if __name__ == "__main__":
    main()
