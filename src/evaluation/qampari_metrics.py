from __future__ import annotations

import re
import string
from typing import Any, Mapping, Sequence

from src.target.answer_parser import parse_answer


def normalize_list_answer(text: str) -> str:
    value = text.casefold()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def parse_list_prediction(text: str) -> list[str]:
    value = parse_answer(text).strip()
    if not value:
        return []
    parts = re.split(r"\s*#\s*|\s*;\s*|\n+", value)
    output = []
    seen = set()
    for part in parts:
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", part).strip()
        normalized = normalize_list_answer(cleaned)
        if normalized and normalized not in seen:
            output.append(cleaned)
            seen.add(normalized)
    return output


def qampari_list_metrics(
    predictions: Sequence[str],
    answer_atoms: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    alias_to_atom: dict[str, int] = {}
    for index, atom in enumerate(answer_atoms):
        aliases = [str(atom["answer_text"]), *(str(value) for value in atom.get("aliases", []))]
        for alias in aliases:
            normalized = normalize_list_answer(alias)
            if normalized and normalized not in alias_to_atom:
                alias_to_atom[normalized] = index
    if not answer_atoms:
        raise ValueError("QAMPARI evaluation requires at least one answer atom")
    matched = set()
    normalized_predictions = []
    for prediction in predictions:
        normalized = normalize_list_answer(prediction)
        if normalized and normalized not in normalized_predictions:
            normalized_predictions.append(normalized)
            if normalized in alias_to_atom:
                matched.add(alias_to_atom[normalized])
    correct = len(matched)
    predicted = len(normalized_predictions)
    gold = len(answer_atoms)
    precision = correct / predicted if predicted else 0.0
    recall = correct / gold
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {
        "correct": correct,
        "predicted": predicted,
        "gold": gold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
