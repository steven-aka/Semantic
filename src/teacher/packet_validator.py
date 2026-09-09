from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from src.data.schemas import SemanticPacket


# Separators belong to a number only when followed by another digit. This keeps
# sentence punctuation ("1999.") from becoming part of the numeric value.
_NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:[$€£])?\d(?:\d|[,.](?=\d))*(?:%|[A-Za-z]+)?")
_ENTITY_RE = re.compile(r"\b(?:[A-Z][\w.-]+(?:\s+[A-Z][\w.-]+)*)\b")
_DOCUMENT_TITLE_RE = re.compile(r"(?im)^Document title:\s*(.+?)\s*$")

_CARDINAL_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30, "fortieth": 40,
    "fiftieth": 50, "sixtieth": 60, "seventieth": 70, "eightieth": 80,
    "ninetieth": 90,
}
_SCALE_WORDS = {
    "hundred": 100, "hundredth": 100,
    "thousand": 1_000, "thousandth": 1_000,
    "million": 1_000_000, "millionth": 1_000_000,
    "billion": 1_000_000_000, "billionth": 1_000_000_000,
}
_NUMBER_WORDS = set(_CARDINAL_WORDS) | set(_ORDINAL_WORDS) | set(_SCALE_WORDS)


def _normalized_values(pattern: re.Pattern[str], text: str) -> set[str]:
    return {match.group(0).lower().replace(",", "") for match in pattern.finditer(text)}


def numeric_strings(text: str) -> tuple[str, ...]:
    """Return the exact numeric strings a packet is allowed to copy."""
    return tuple(dict.fromkeys(match.group(0) for match in _NUMBER_RE.finditer(text)))


def document_titles(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.group(1).strip() for match in _DOCUMENT_TITLE_RE.finditer(text)))


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def normalize_document_identity(text: str) -> str:
    """Remove source-format markup while preserving the displayed identity."""
    value = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", text)
    value = re.sub(r"[*_]+", "", value)
    return _normalized_text(value)


def _word_number_value(words: list[str]) -> int:
    total = 0
    current = 0
    for word in words:
        if word == "and":
            continue
        if word in _CARDINAL_WORDS:
            current += _CARDINAL_WORDS[word]
        elif word in _ORDINAL_WORDS:
            current += _ORDINAL_WORDS[word]
        else:
            scale = _SCALE_WORDS[word]
            if scale == 100:
                current = max(current, 1) * scale
            else:
                total += max(current, 1) * scale
                current = 0
    return total + current


def _digit_concepts(text: str) -> set[str]:
    concepts: set[str] = set()
    pattern = re.compile(
        r"(?<![\w\d])[+-]?(?:[$€£])?\d(?:[\d,]*\d)?(?:\.\d+)?(?:st|nd|rd|th)?"
        r"|(?<=[A-Za-z])\d(?:[\d,]*\d)?(?:\.\d+)?(?:st|nd|rd|th)?",
        re.I,
    )
    for match in pattern.finditer(text):
        value = match.group(0).lower().replace(",", "")
        value = re.sub(r"^(?:[$€£])", "", value)
        value = re.sub(r"(?:st|nd|rd|th)$", "", value)
        concepts.add(value)
    return concepts


def numeric_concepts(text: str) -> set[str]:
    """Canonical numeric values stated as digits, cardinals, or ordinals.

    This is used on source text so surface-equivalent forms such as ``ninth``
    and ``9th`` map to the same value. Compact forms such as ``1h21m`` are
    scanned for each numeric component.
    """
    concepts = _digit_concepts(text)

    run: list[str] = []
    for match in re.finditer(r"[A-Za-z]+", text.casefold()):
        word = match.group(0)
        if word in _NUMBER_WORDS or (word == "and" and run):
            run.append(word)
            continue
        if run:
            while run and run[-1] == "and":
                run.pop()
            if run:
                concepts.add(str(_word_number_value(run)))
            run = []
    while run and run[-1] == "and":
        run.pop()
    if run:
        concepts.add(str(_word_number_value(run)))
    return concepts


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def validate_packet(packet: SemanticPacket, max_gist_ratio: float = 0.45) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if not packet.source.strip() or not packet.gist.strip():
        errors.append("source and gist must be non-empty")
    if packet.source_tokens <= 0:
        errors.append("source token count must be positive")
    elif packet.gist_tokens / packet.source_tokens > max_gist_ratio:
        errors.append(
            f"gist exceeds {max_gist_ratio:.0%} of source tokens "
            f"({packet.gist_tokens}/{packet.source_tokens})"
        )
    if packet.gist_tokens >= packet.source_tokens:
        errors.append("gist must be shorter than source")
    if packet.residual.strip():
        repetition = SequenceMatcher(None, packet.gist.lower(), packet.residual.lower()).ratio()
        if repetition > 0.80:
            errors.append("residual excessively repeats gist")
    else:
        warnings.append("residual is empty")
    source_numbers = numeric_concepts(packet.source)
    # Keep the original guard's scope: reject explicit output digits absent
    # from the source. Source number words are additionally accepted as the
    # same numeric value; ordinary prose such as "one of" is not a false alarm.
    output_numbers = _digit_concepts(packet.gist + " " + packet.residual)
    unexpected_numbers = output_numbers - source_numbers
    if unexpected_numbers:
        errors.append(
            "packet contains numbers absent from source: "
            + ", ".join(sorted(unexpected_numbers))
        )
    combined = normalize_document_identity(packet.gist + " " + packet.residual)
    missing_titles = [
        title
        for title in document_titles(packet.source)
        if normalize_document_identity(title) not in combined
    ]
    if missing_titles:
        errors.append("packet omits source document titles: " + ", ".join(missing_titles))
    source_entities = _normalized_values(_ENTITY_RE, packet.source)
    output_entities = _normalized_values(_ENTITY_RE, packet.gist + " " + packet.residual)
    unseen_entities = output_entities - source_entities
    if unseen_entities:
        warnings.append("possible entities absent from source: " + ", ".join(sorted(unseen_entities)))
    return ValidationResult(not errors, tuple(errors), tuple(warnings))
