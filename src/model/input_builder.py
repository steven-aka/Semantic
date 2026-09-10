from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class AtomicModelInput:
    input_ids: list[int]
    packet_spans: list[tuple[int, int]]


def build_atomic_model_input(
    tokenizer: Any,
    question: str,
    packet_texts: Sequence[str],
    *,
    max_length: int = 4096,
) -> AtomicModelInput:
    """Tokenize an evidence set while retaining exact source-packet spans."""
    if not question.strip() or not packet_texts or any(not text.strip() for text in packet_texts):
        raise ValueError("a question and non-empty packet texts are required")

    def encode(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False).input_ids)

    input_ids = encode(f"Question:\n{question}\n\nEvidence packets:")
    spans: list[tuple[int, int]] = []
    for packet_id, text in enumerate(packet_texts):
        input_ids.extend(encode(f"\n\nPacket {packet_id}:\n"))
        start = len(input_ids)
        input_ids.extend(encode(text))
        end = len(input_ids)
        if start == end:
            raise ValueError(f"packet {packet_id} tokenized to an empty span")
        spans.append((start, end))
    input_ids.extend(encode("\n\nEnd of evidence."))
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        input_ids.append(int(eos))
    if len(input_ids) > max_length:
        raise ValueError(
            f"atomic policy input has {len(input_ids)} tokens, exceeds {max_length}; "
            "scientific inputs are not silently truncated"
        )
    return AtomicModelInput(input_ids=input_ids, packet_spans=spans)


def collate_atomic_inputs(
    rows: Sequence[AtomicModelInput], pad_token_id: int
) -> tuple[Any, Any, list[list[tuple[int, int]]]]:
    import torch

    if not rows:
        raise ValueError("cannot collate an empty batch")
    width = max(len(row.input_ids) for row in rows)
    input_ids = torch.full((len(rows), width), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for index, row in enumerate(rows):
        length = len(row.input_ids)
        input_ids[index, :length] = torch.tensor(row.input_ids, dtype=torch.long)
        attention_mask[index, :length] = 1
    return input_ids, attention_mask, [row.packet_spans for row in rows]
