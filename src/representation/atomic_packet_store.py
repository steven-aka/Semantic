from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.data.schemas import AtomicPacket, read_jsonl, write_jsonl


class AtomicPacketStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, example_id: str) -> Path:
        if not example_id or "/" in example_id or ".." in example_id:
            raise ValueError("unsafe example_id")
        return self.directory / f"{example_id}.jsonl"

    def put(self, example_id: str, packets: Sequence[AtomicPacket]) -> Path:
        if any(packet.example_id != example_id for packet in packets):
            raise ValueError("all packets must match example_id")
        if [packet.packet_id for packet in packets] != list(range(len(packets))):
            raise ValueError("atomic packet ids must be contiguous")
        path = self.path(example_id)
        write_jsonl(path, packets)
        return path

    def get(self, example_id: str) -> list[AtomicPacket]:
        packets = list(read_jsonl(self.path(example_id), AtomicPacket))
        if [packet.packet_id for packet in packets] != list(range(len(packets))):
            raise ValueError(f"non-contiguous atomic packets for {example_id}")
        return packets

