from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    special_answers = {"yes", "no", "noanswer"}
    if (
        normalized_prediction in special_answers or normalized_gold in special_answers
    ) and normalized_prediction != normalized_gold:
        return 0.0
    predicted = normalized_prediction.split()
    expected = normalized_gold.split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def supporting_fact_recall(states: tuple[int, ...], supporting: list[bool]) -> float:
    if len(states) != len(supporting):
        raise ValueError("states and supporting flags must have equal length")
    indices = [index for index, flag in enumerate(supporting) if flag]
    if not indices:
        return 1.0
    return sum(states[index] >= 1 for index in indices) / len(indices)
