from __future__ import annotations

import re
from typing import Any


_FALLBACK_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


class WhitespaceTokenizer:
    """Dependency-free tokenizer for tests only, never for reported rates."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return _FALLBACK_TOKEN_RE.findall(text)

    def decode(self, tokens: list[str], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(tokens)


def encode_text(tokenizer: Any, text: str) -> list[Any]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return list(ids)


def count_tokens(tokenizer: Any, text: str) -> int:
    return len(encode_text(tokenizer, text))


def load_tokenizer(model_name: str, revision: str = "main") -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for scientific token accounting; "
            "WhitespaceTokenizer is only valid for code tests"
        ) from exc
    return AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)

