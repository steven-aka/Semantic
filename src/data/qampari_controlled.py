from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data.schemas import QAExample, SemanticUnit, write_jsonl
from src.evaluation.qampari_metrics import normalize_list_answer
from src.representation.token_counter import count_tokens, load_tokenizer
from src.reproducibility import sha256, write_metadata


def _shortest_proof(answer: Mapping[str, Any], tokenizer: Any) -> str:
    proofs = [
        str(proof.get("proof_text", "")).strip()
        for proof in answer.get("proof", [])
        if str(proof.get("proof_text", "")).strip()
    ]
    if not proofs:
        raise ValueError("answer atom has no non-empty proof")
    return min(proofs, key=lambda value: (count_tokens(tokenizer, value), value))


_QUESTION_STOPWORDS = {
    "a", "an", "and", "are", "at", "by", "did", "do", "does", "for", "from",
    "how", "in", "is", "of", "on", "or", "the", "to", "was", "were", "what",
    "when", "where", "which", "who", "with",
}


def select_query_sentence(proof: str, question: str) -> str:
    sentences = [
        value.strip()
        for value in re.split(r"(?<=[.!?])\s+|\n+", proof.strip())
        if value.strip()
    ]
    if not sentences:
        raise ValueError("proof contains no sentence")
    query_terms = {
        token
        for token in re.findall(r"[a-z0-9]+", question.casefold())
        if token not in _QUESTION_STOPWORDS
    }
    scores = []
    for index, sentence in enumerate(sentences):
        sentence_terms = set(re.findall(r"[a-z0-9]+", sentence.casefold()))
        scores.append((len(query_terms & sentence_terms), -index, sentence))
    return max(scores)[2]


def _answer_atoms(
    row: Mapping[str, Any], tokenizer: Any, n_answers: int
) -> list[dict[str, Any]]:
    atoms = []
    seen = set()
    for answer in row.get("answer_list", []):
        canonical = str(answer.get("answer_text", "")).strip()
        normalized = normalize_list_answer(canonical)
        if not normalized or normalized in seen:
            continue
        source_proof = _shortest_proof(answer, tokenizer)
        proof = select_query_sentence(source_proof, str(row["question_text"]))
        atoms.append(
            {
                "answer_text": canonical,
                "aliases": sorted(
                    {
                        alias.strip()
                        for alias in (str(value) for value in answer.get("aliases", []))
                        if alias.strip()
                    }
                ),
                "proof": proof,
                "source_proof": source_proof,
                "answer_url": str(answer.get("answer_url", "")),
            }
        )
        seen.add(normalized)
        if len(atoms) == n_answers:
            break
    return atoms


def _render_group(atoms: Sequence[Mapping[str, Any]]) -> str:
    return "\n\n".join(
        f"Document title: {atom['answer_text']}\n{atom['proof']}" for atom in atoms
    )


def build_controlled_examples(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    n_examples: int,
    n_answers: int = 10,
    max_unit_tokens: int = 256,
    seed: int = 20260908,
) -> tuple[list[QAExample], list[dict[str, Any]]]:
    if n_answers != 10:
        raise ValueError("the first controlled protocol requires exactly ten answer atoms")
    if n_examples <= 0:
        raise ValueError("n_examples must be positive")
    eligible = []
    for row in rows:
        try:
            atoms = _answer_atoms(row, tokenizer, n_answers)
        except ValueError:
            continue
        if len(atoms) != n_answers:
            continue
        groups = [atoms[index : index + 2] for index in range(0, n_answers, 2)]
        rendered = [_render_group(group) for group in groups]
        if any(count_tokens(tokenizer, value) > max_unit_tokens for value in rendered):
            continue
        eligible.append((row, atoms, rendered))
    if len(eligible) < n_examples + 1:
        raise ValueError("not enough eligible rows to construct examples and distractors")

    # A stable hash order prevents source-file ordering from becoming a tuning knob.
    eligible.sort(
        key=lambda item: hashlib.sha256(
            f"{seed}:{item[0]['qid']}".encode("utf-8")
        ).hexdigest()
    )
    examples = []
    annotations = []
    for index, (row, atoms, relevant_groups) in enumerate(eligible[:n_examples]):
        distractor_source = eligible[(index + n_examples) % len(eligible)][1][:2]
        distractor = _render_group(distractor_source)
        if count_tokens(tokenizer, distractor) > max_unit_tokens:
            raise ValueError("deterministic distractor exceeds max_unit_tokens")
        unit_rows = [
            {"text": text, "answer_indices": [2 * group, 2 * group + 1]}
            for group, text in enumerate(relevant_groups)
        ]
        unit_rows.append({"text": distractor, "answer_indices": []})
        rng = random.Random(f"{seed}:{row['qid']}")
        rng.shuffle(unit_rows)
        units = [
            SemanticUnit(
                unit_id=unit_id,
                text=unit["text"],
                supporting=False,
                title=None,
            )
            for unit_id, unit in enumerate(unit_rows)
        ]
        example_id = str(row["qid"])
        examples.append(
            QAExample(
                example_id=example_id,
                dataset="qampari_controlled_dev",
                question=str(row["question_text"]).strip(),
                answer=" # ".join(atom["answer_text"] for atom in atoms),
                context="\n\n".join(unit.text for unit in units),
                units=units,
            )
        )
        annotations.append(
            {
                "example_id": example_id,
                "answer_atoms": atoms,
                "unit_answer_indices": [unit["answer_indices"] for unit in unit_rows],
                "n_answer_atoms": n_answers,
                "n_relevant_units": 5,
                "n_distractor_units": 1,
            }
        )
    return examples, annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the controlled QAMPARI M0 development set")
    parser.add_argument("--input", required=True)
    parser.add_argument("--examples-output", required=True)
    parser.add_argument("--annotations-output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--tokenizer", default="models/Qwen3-8B")
    parser.add_argument("--n-examples", type=int, default=30)
    parser.add_argument("--n-answers", type=int, default=10)
    parser.add_argument("--max-unit-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).open(encoding="utf-8") if line.strip()]
    tokenizer = load_tokenizer(args.tokenizer)
    examples, annotations = build_controlled_examples(
        rows,
        tokenizer,
        n_examples=args.n_examples,
        n_answers=args.n_answers,
        max_unit_tokens=args.max_unit_tokens,
        seed=args.seed,
    )
    write_jsonl(args.examples_output, examples)
    write_jsonl(args.annotations_output, annotations)
    write_metadata(
        args.metadata_output,
        {
            "stage": "m0_qampari_controlled_development_only",
            "source_split": "dev",
            "input_sha256": sha256(args.input),
            "examples": len(examples),
            "answer_atoms_per_example": args.n_answers,
            "units_per_example": 6,
            "relevant_units": 5,
            "distractor_units": 1,
            "max_unit_tokens": args.max_unit_tokens,
            "proof_normalization": "single source sentence with maximum question-term overlap; source-order tie break",
            "seed": args.seed,
            "locked_model_outputs_observed": False,
        },
    )


if __name__ == "__main__":
    main()
