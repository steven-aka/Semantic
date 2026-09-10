from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from src.data.schemas import AtomicPacket, QAExample, read_jsonl
from src.representation.atomic_packet_store import AtomicPacketStore
from src.representation.token_counter import count_tokens, load_tokenizer
from src.reproducibility import experiment_metadata, write_metadata


def build_atomic_packets(
    example: QAExample, tokenizer: Any, *, blocks_per_unit: int = 2
) -> list[AtomicPacket]:
    packets: list[AtomicPacket] = []
    for unit in example.units:
        blocks = [block.strip() for block in unit.text.split("\n\n")]
        if len(blocks) != blocks_per_unit or any(not block for block in blocks):
            raise ValueError(
                f"{example.example_id}/{unit.unit_id}: require exactly "
                f"{blocks_per_unit} non-empty source blocks"
            )
        for block_index, block in enumerate(blocks):
            packets.append(
                AtomicPacket(
                    example_id=example.example_id,
                    packet_id=len(packets),
                    source_unit_id=unit.unit_id,
                    source_block_index=block_index,
                    text=block,
                    tokens=count_tokens(tokenizer, block),
                )
            )
    reconstructed = "\n\n".join(packet.text for packet in packets)
    if reconstructed != example.context.strip():
        raise ValueError(f"{example.example_id}: atomic packets do not reconstruct context")
    return packets


def validate_atomic_packets(
    example: QAExample, packets: Sequence[AtomicPacket], tokenizer: Any
) -> list[str]:
    errors: list[str] = []
    if [packet.packet_id for packet in packets] != list(range(len(packets))):
        errors.append("packet ids are not contiguous")
    if any(packet.example_id != example.example_id for packet in packets):
        errors.append("example id mismatch")
    coordinates = [
        (packet.source_unit_id, packet.source_block_index) for packet in packets
    ]
    if coordinates != sorted(coordinates) or len(set(coordinates)) != len(coordinates):
        errors.append("source coordinates are not unique and ordered")
    if any(packet.tokens != count_tokens(tokenizer, packet.text) for packet in packets):
        errors.append("packet token count mismatch")
    if "\n\n".join(packet.text for packet in packets) != example.context.strip():
        errors.append("packets do not reconstruct source context")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build source-ordered lossless binary atomic packets"
    )
    parser.add_argument("--examples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", default="models/Qwen3-8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--blocks-per-unit", type=int, default=2)
    args = parser.parse_args()

    examples = list(read_jsonl(args.examples, QAExample))
    tokenizer = load_tokenizer(args.tokenizer, args.revision)
    store = AtomicPacketStore(args.output_dir)
    packet_count = 0
    for example in examples:
        packets = build_atomic_packets(
            example, tokenizer, blocks_per_unit=args.blocks_per_unit
        )
        store.put(example.example_id, packets)
        packet_count += len(packets)
    write_metadata(
        Path(args.output_dir) / "metadata.json",
        experiment_metadata(
            stage="m0_atomic_binary_lossless_packetization",
            input_examples=args.examples,
            examples=len(examples),
            packets=packet_count,
            packets_per_example=(packet_count // len(examples) if examples else 0),
            blocks_per_unit=args.blocks_per_unit,
            tokenizer=args.tokenizer,
            tokenizer_revision=args.revision,
            packetization="original source blocks; one binary reveal state per block",
            render_order="source unit then source block",
            reads_question=False,
            reads_gold_annotations=False,
        ),
    )


if __name__ == "__main__":
    main()
