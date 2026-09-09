from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, TypeVar


@dataclass(frozen=True)
class SemanticUnit:
    unit_id: int
    text: str
    supporting: bool = False
    title: str | None = None
    # Optional source coordinate.  Label-free sentence-atomic pilots use this
    # to audit fact mapping without consulting supporting labels during unit
    # construction.  Older artifacts intentionally deserialize with None.
    source_sentence_id: int | None = None

    def __post_init__(self) -> None:
        if self.unit_id < 0:
            raise ValueError("unit_id must be non-negative")
        if not self.text.strip():
            raise ValueError("semantic unit text must be non-empty")


@dataclass(frozen=True)
class QAExample:
    example_id: str
    dataset: str
    question: str
    answer: str
    context: str
    units: list[SemanticUnit] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.example_id or not self.question.strip():
            raise ValueError("example_id and question are required")
        ids = [unit.unit_id for unit in self.units]
        if ids != list(range(len(ids))):
            raise ValueError("unit_id values must be contiguous and preserve source order")


@dataclass(frozen=True)
class SemanticPacket:
    example_id: str
    unit_id: int
    source: str
    gist: str
    residual: str
    source_tokens: int
    gist_tokens: int
    residual_tokens: int

    def __post_init__(self) -> None:
        if self.unit_id < 0:
            raise ValueError("unit_id must be non-negative")
        for name in ("source_tokens", "gist_tokens", "residual_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class ExactSearchResult:
    example_id: str
    state: tuple[int, ...]
    tokens: int
    prediction: str
    answer_em: float
    answer_f1: float
    fact_recall: float
    fidelity: float

    def __post_init__(self) -> None:
        if any(value not in (0, 1, 2) for value in self.state):
            raise ValueError("state values must be 0, 1, or 2")
        if self.tokens < 0:
            raise ValueError("tokens must be non-negative")


T = TypeVar("T")


def _construct(cls: type[T], row: Mapping[str, Any]) -> T:
    data = dict(row)
    if cls is QAExample:
        data["units"] = [SemanticUnit(**unit) for unit in data.get("units", [])]
    elif cls is ExactSearchResult:
        data["state"] = tuple(data["state"])
    return cls(**data)


def read_jsonl(path: str | Path, cls: type[T] | None = None) -> Iterator[T | dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            yield _construct(cls, row) if cls else row


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                value = asdict(row) if hasattr(row, "__dataclass_fields__") else row
                handle.write(
                    json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
