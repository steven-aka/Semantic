from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from src.data.schemas import SemanticPacket, read_jsonl, write_jsonl


class PacketStore:
    """JSONL packet store keyed by example id."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, example_id: str) -> Path:
        if not example_id or "/" in example_id or ".." in example_id:
            raise ValueError("unsafe example_id")
        return self.directory / f"{example_id}.jsonl"

    def put(self, example_id: str, packets: Sequence[SemanticPacket]) -> Path:
        if any(packet.example_id != example_id for packet in packets):
            raise ValueError("all packets must match example_id")
        path = self.path(example_id)
        write_jsonl(path, packets)
        return path

    def get(self, example_id: str) -> list[SemanticPacket]:
        return list(read_jsonl(self.path(example_id), SemanticPacket))

    def contains(self, example_id: str) -> bool:
        return self.path(example_id).exists()

