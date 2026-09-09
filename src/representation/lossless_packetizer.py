from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from src.data.schemas import QAExample, SemanticPacket, read_jsonl, write_jsonl
from src.representation.token_counter import count_tokens, load_tokenizer
from src.reproducibility import experiment_metadata, write_metadata


def partition_unit(
    example_id: str, unit_id: int, source: str, tokenizer: Any
) -> SemanticPacket:
    """Partition a two-document unit into lossless additive source blocks."""
    blocks = source.split("\n\n")
    if len(blocks) != 2 or any(not block.strip() for block in blocks):
        raise ValueError("lossless M0 packetization requires exactly two source blocks")
    lengths = [count_tokens(tokenizer, block) for block in blocks]
    gist_index = min(range(2), key=lambda index: (lengths[index], index))
    residual_index = 1 - gist_index
    return SemanticPacket(
        example_id=example_id,
        unit_id=unit_id,
        source=source,
        gist=blocks[gist_index].strip(),
        residual=blocks[residual_index].strip(),
        source_tokens=count_tokens(tokenizer, source),
        gist_tokens=lengths[gist_index],
        residual_tokens=lengths[residual_index],
    )


def validate_lossless_partition(packet: SemanticPacket, tokenizer: Any) -> list[str]:
    errors: list[str] = []
    blocks = packet.source.split("\n\n")
    if len(blocks) != 2:
        return ["source does not contain exactly two blocks"]
    if not (
        (packet.gist == blocks[0].strip() and packet.residual == blocks[1].strip())
        or (packet.gist == blocks[1].strip() and packet.residual == blocks[0].strip())
    ):
        errors.append("gist and residual are not a disjoint exhaustive source partition")
    expected = {
        "source_tokens": count_tokens(tokenizer, packet.source),
        "gist_tokens": count_tokens(tokenizer, packet.gist),
        "residual_tokens": count_tokens(tokenizer, packet.residual),
    }
    for name, value in expected.items():
        if getattr(packet, name) != value:
            errors.append(f"{name} mismatch")
    if packet.source_tokens and packet.gist_tokens / packet.source_tokens > 0.5:
        errors.append("shorter source block exceeds half of source tokens")
    return errors


def build_lossless_packets(
    examples: Sequence[QAExample], tokenizer: Any
) -> dict[str, list[SemanticPacket]]:
    output: dict[str, list[SemanticPacket]] = {}
    for example in examples:
        packets = [
            partition_unit(example.example_id, unit.unit_id, unit.text, tokenizer)
            for unit in example.units
        ]
        for packet in packets:
            errors = validate_lossless_partition(packet, tokenizer)
            if errors:
                raise ValueError(
                    f"invalid lossless partition {packet.example_id}/{packet.unit_id}: "
                    + "; ".join(errors)
                )
        output[example.example_id] = packets
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic lossless additive packets for QAMPARI M0"
    )
    parser.add_argument("--examples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", default="models/Qwen3-8B")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    examples = list(read_jsonl(args.examples, QAExample))
    tokenizer = load_tokenizer(args.tokenizer, args.revision)
    packets_by_id = build_lossless_packets(examples, tokenizer)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for example in examples:
        write_jsonl(output_dir / f"{example.example_id}.jsonl", packets_by_id[example.example_id])
    write_metadata(
        output_dir / "metadata.json",
        experiment_metadata(
            stage="m0_qampari_lossless_additive_source_partition",
            scientific_status="development_only_not_a_locked_result",
            input_examples=args.examples,
            tokenizer=args.tokenizer,
            tokenizer_revision=args.revision,
            packetization=(
                "split each two-document unit at its original block boundary; "
                "shorter target-tokenizer block is gist, other block is residual; "
                "source-order tie break"
            ),
            reads_question=False,
            reads_gold_annotations=False,
            maximum_gist_source_ratio=0.5,
        ),
    )


if __name__ == "__main__":
    main()
