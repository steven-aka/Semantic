from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

from src.data.schemas import QAExample, SemanticUnit, write_jsonl
from src.evaluation.qampari_metrics import normalize_list_answer
from src.representation.token_counter import count_tokens, load_tokenizer
from src.reproducibility import sha256, write_metadata


_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "the",
    "to", "was", "were", "what", "when", "where", "which", "who", "with",
}
_GENERIC_LIST_TERMS = {"example", "examples", "list", "name", "names"}


def wiki_display(text: str) -> str:
    value = html.unescape(str(text)).strip()

    def replace(match: re.Match[str]) -> str:
        return match.group(1).rsplit("|", 1)[-1].strip()

    value = re.sub(r"\[\[([^\[\]]+)\]\]", replace, value)
    return " ".join(value.replace("_", " ").split())


def wiki_url_title(url: str) -> str:
    path = unquote(urlparse(str(url)).path)
    return wiki_display(path.rsplit("/", 1)[-1] if path else "")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _relation_terms(row: Mapping[str, Any]) -> set[str]:
    entity_terms = set()
    for entity in row.get("entities", []):
        entity_terms.update(_tokens(str(entity.get("entity_text", ""))))
        for alias in entity.get("aliases", []):
            entity_terms.update(_tokens(str(alias)))
    return {
        token for token in _tokens(str(row["question_text"]))
        if token not in _STOPWORDS
        and token not in _GENERIC_LIST_TERMS
        and token not in entity_terms
        and not token.isdigit()
    }


def _numbers(text: str) -> list[float]:
    values = []
    for match in re.finditer(r"(?<![a-z0-9])\d[\d,]*(?:\.\d+)?", text.casefold()):
        values.append(float(match.group(0).replace(",", "")))
    return values


def numeric_constraint_passes(question: str, proof: str) -> bool:
    q = question.casefold()
    p_values = _numbers(proof)
    patterns = (
        (r"\bafter\s+(\d{3,4})\b", lambda value, bound: value > bound),
        (r"\bbefore\s+(\d{3,4})\b", lambda value, bound: value < bound),
        (r"\b(?:larger|longer|greater|more)\s+than\s+([\d,]+(?:\.\d+)?)", lambda value, bound: value > bound),
        (r"\b(?:smaller|shorter|less)\s+than\s+([\d,]+(?:\.\d+)?)", lambda value, bound: value < bound),
    )
    for pattern, relation in patterns:
        match = re.search(pattern, q)
        if match:
            bound = float(match.group(1).replace(",", ""))
            return any(relation(value, bound) for value in p_values)
    return True


def _same_article(answer: Mapping[str, Any], proof: Mapping[str, Any]) -> bool:
    answer_title = normalize_list_answer(wiki_url_title(str(answer.get("answer_url", ""))))
    proof_title = normalize_list_answer(wiki_url_title(str(proof.get("found_in_url", ""))))
    return bool(answer_title and proof_title and answer_title == proof_title)


def certified_atom(row: Mapping[str, Any], answer: Mapping[str, Any], tokenizer: Any) -> dict[str, Any] | None:
    canonical = wiki_display(str(answer.get("answer_text", "")))
    if not normalize_list_answer(canonical):
        return None
    relation_terms = _relation_terms(row)
    candidates = []
    for proof in answer.get("proof", []):
        proof_text = str(proof.get("proof_text", "")).strip()
        if not proof_text or not _same_article(answer, proof):
            continue
        overlap = len(relation_terms & _tokens(proof_text))
        if relation_terms and overlap == 0:
            continue
        if not numeric_constraint_passes(str(row["question_text"]), proof_text):
            continue
        candidates.append((-overlap, count_tokens(tokenizer, proof_text), proof_text, proof))
    if not candidates:
        return None
    _, _, proof_text, proof_row = min(candidates, key=lambda value: value[:3])
    aliases = {
        canonical,
        wiki_display(str(answer.get("answer_text", ""))),
        wiki_url_title(str(answer.get("answer_url", ""))),
        *(wiki_display(str(value)) for value in answer.get("aliases", [])),
    }
    aliases = {value for value in aliases if normalize_list_answer(value)}
    return {
        "answer_text": canonical,
        "aliases": sorted(aliases),
        "proof": proof_text,
        "source_proof": proof_text,
        "answer_url": str(answer.get("answer_url", "")),
        "proof_url": str(proof_row.get("found_in_url", "")),
        "certificate": {
            "canonical_identity": True,
            "same_article_provenance": True,
            "relation_term_overlap": sorted(relation_terms & _tokens(proof_text)),
            "numeric_constraint_pass": True,
            "full_proof_preserved": True,
        },
    }


def _render_atom(atom: Mapping[str, Any]) -> str:
    return f"Document title: {atom['answer_text']}\nSource evidence: {atom['proof']}"


def build_ceiling_candidates(
    rows: Sequence[Mapping[str, Any]], tokenizer: Any, *, n_examples: int,
    excluded_ids: set[str] | None = None, n_answers: int = 10,
    max_unit_tokens: int = 512, seed: int = 20260909,
) -> tuple[list[QAExample], list[dict[str, Any]], dict[str, int]]:
    if n_answers != 10:
        raise ValueError("ceiling-v2 protocol requires exactly ten answer atoms")
    excluded_ids = excluded_ids or set()
    eligible = []
    rejections = {"excluded_id": 0, "fewer_than_ten_certified_atoms": 0, "unit_too_long": 0}
    for row in rows:
        qid = str(row["qid"])
        if qid in excluded_ids:
            rejections["excluded_id"] += 1
            continue
        atoms, seen = [], set()
        for answer in row.get("answer_list", []):
            atom = certified_atom(row, answer, tokenizer)
            if atom is None:
                continue
            normalized = normalize_list_answer(str(atom["answer_text"]))
            if normalized in seen:
                continue
            atoms.append(atom)
            seen.add(normalized)
            if len(atoms) == n_answers:
                break
        if len(atoms) != n_answers:
            rejections["fewer_than_ten_certified_atoms"] += 1
            continue
        groups = [atoms[index:index + 2] for index in range(0, n_answers, 2)]
        rendered = ["\n\n".join(_render_atom(atom) for atom in group) for group in groups]
        if any(count_tokens(tokenizer, value) > max_unit_tokens for value in rendered):
            rejections["unit_too_long"] += 1
            continue
        eligible.append((row, atoms, rendered))
    eligible.sort(key=lambda item: hashlib.sha256(f"{seed}:{item[0]['qid']}".encode()).hexdigest())
    if len(eligible) < n_examples + 1:
        raise ValueError(f"only {len(eligible)} evidence-certified examples; require {n_examples + 1}")
    examples, annotations = [], []
    for index, (row, atoms, relevant_groups) in enumerate(eligible[:n_examples]):
        distractor_atoms = eligible[(index + n_examples) % len(eligible)][1][:2]
        distractor = "\n\n".join(_render_atom(atom) for atom in distractor_atoms)
        if count_tokens(tokenizer, distractor) > max_unit_tokens:
            raise ValueError("deterministic distractor exceeds max_unit_tokens")
        unit_rows = [
            {"text": text, "answer_indices": [2 * group, 2 * group + 1]}
            for group, text in enumerate(relevant_groups)
        ] + [{"text": distractor, "answer_indices": []}]
        random.Random(f"{seed}:{row['qid']}").shuffle(unit_rows)
        units = [SemanticUnit(unit_id=i, text=unit["text"], supporting=False) for i, unit in enumerate(unit_rows)]
        examples.append(QAExample(
            example_id=str(row["qid"]), dataset="qampari_ceiling_v2_development",
            question=str(row["question_text"]).strip(),
            answer=" # ".join(str(atom["answer_text"]) for atom in atoms),
            context="\n\n".join(unit.text for unit in units), units=units,
        ))
        annotations.append({
            "example_id": str(row["qid"]), "answer_atoms": atoms,
            "unit_answer_indices": [unit["answer_indices"] for unit in unit_rows],
            "n_answer_atoms": n_answers, "n_relevant_units": 5,
            "n_distractor_units": 1, "evidence_certificate_pass": True,
        })
    return examples, annotations, rejections


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evidence-closed QAMPARI ceiling-v2 candidates")
    parser.add_argument("--input", required=True)
    parser.add_argument("--examples-output", required=True)
    parser.add_argument("--annotations-output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--exclude-examples", action="append", default=[])
    parser.add_argument("--tokenizer", default="models/Qwen3-8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--n-examples", type=int, default=120)
    parser.add_argument("--max-unit-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).open(encoding="utf-8") if line.strip()]
    excluded_ids = set()
    for path in args.exclude_examples:
        excluded_ids.update(str(json.loads(line)["example_id"]) for line in Path(path).open(encoding="utf-8"))
    tokenizer = load_tokenizer(args.tokenizer, args.revision)
    examples, annotations, rejections = build_ceiling_candidates(
        rows, tokenizer, n_examples=args.n_examples, excluded_ids=excluded_ids,
        max_unit_tokens=args.max_unit_tokens, seed=args.seed,
    )
    write_jsonl(args.examples_output, examples)
    write_jsonl(args.annotations_output, annotations)
    write_metadata(args.metadata_output, {
        "stage": "m0_qampari_ceiling_v2_candidates_before_target_inference",
        "source_split": "dev", "input_sha256": sha256(args.input),
        "examples": len(examples), "answer_atoms_per_example": 10,
        "units_per_example": 6, "max_unit_tokens": args.max_unit_tokens,
        "seed": args.seed, "excluded_examples": len(excluded_ids),
        "rejection_counts": rejections,
        "certificate": "display identity; same-article proof; relation overlap; numeric bound; full proof",
        "target_outputs_observed_before_candidate_freeze": False,
        "examples_sha256": sha256(args.examples_output),
        "annotations_sha256": sha256(args.annotations_output),
    })


if __name__ == "__main__":
    main()
