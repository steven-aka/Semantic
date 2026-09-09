from __future__ import annotations

import re


_PREFIX_RE = re.compile(r"^(?:final answer|answer)\s*:\s*", re.IGNORECASE)
_BOOLEAN_SENTENCE_RE = re.compile(r"^(yes|no)\s*[,;:!.?]", re.IGNORECASE)
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def parse_answer(text: str) -> str:
    """Normalize model wrapping while retaining the answer text itself."""
    value = text.strip()
    if "</think>" in value:
        value = value.rsplit("</think>", 1)[-1].strip()
    tagged = _ANSWER_TAG_RE.search(value)
    if tagged:
        return tagged.group(1).strip()
    value = _PREFIX_RE.sub("", value)
    value = value.splitlines()[0].strip() if value else ""
    boolean = _BOOLEAN_SENTENCE_RE.match(value)
    if boolean:
        return boolean.group(1).lower()
    return value
