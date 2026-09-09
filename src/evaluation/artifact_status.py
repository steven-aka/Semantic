from __future__ import annotations

import argparse
import json
import re
from itertools import product
from pathlib import Path
from typing import Any

from src.data.schemas import ExactSearchResult, QAExample, SemanticPacket, read_jsonl
from src.teacher.packet_validator import validate_packet


def _safe_id(example_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", example_id)


def packet_status(examples_path: str | Path, packet_dir: str | Path) -> dict[str, Any]:
    examples = list(read_jsonl(examples_path, QAExample))
    root = Path(packet_dir)
    errors: list[str] = []
    completed = 0
    for example in examples:
        path = root / f"{_safe_id(example.example_id)}.jsonl"
        if not path.exists():
            errors.append(f"missing packet file: {path}")
            continue
        try:
            packets = list(read_jsonl(path, SemanticPacket))
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"invalid packet file {path}: {exc}")
            continue
        if len(packets) != len(example.units):
            errors.append(
                f"packet count mismatch for {example.example_id}: "
                f"{len(packets)} != {len(example.units)}"
            )
            continue
        valid = True
        for unit, packet in zip(example.units, packets):
            if (
                packet.example_id != example.example_id
                or packet.unit_id != unit.unit_id
                or packet.source != unit.text
            ):
                errors.append(f"packet/source mismatch: {example.example_id}/{unit.unit_id}")
                valid = False
                break
            validation = validate_packet(packet)
            if not validation.valid:
                errors.append(
                    f"invalid packet {example.example_id}/{unit.unit_id}: "
                    + "; ".join(validation.errors)
                )
                valid = False
                break
        if valid:
            completed += 1
    return {
        "kind": "packets",
        "complete": completed == len(examples) and not errors,
        "completed_examples": completed,
        "expected_examples": len(examples),
        "errors": errors,
    }


def exact_status(examples_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    examples = list(read_jsonl(examples_path, QAExample))
    root = Path(output_dir)
    errors: list[str] = []
    completed = 0
    for example in examples:
        path = root / f"{example.example_id}.jsonl"
        if not path.exists():
            errors.append(f"missing exact-search file: {path}")
            continue
        try:
            rows = list(read_jsonl(path, ExactSearchResult))
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"invalid exact-search file {path}: {exc}")
            continue
        row_errors = validate_exact_rows(example, rows)
        if row_errors:
            errors.append(
                f"invalid exact search for {example.example_id}: " + "; ".join(row_errors)
            )
            continue
        completed += 1
    return {
        "kind": "exact_search",
        "complete": completed == len(examples) and not errors,
        "completed_examples": completed,
        "expected_examples": len(examples),
        "errors": errors,
    }


def validate_exact_rows(
    example: QAExample, rows: list[ExactSearchResult]
) -> list[str]:
    errors: list[str] = []
    expected_states = set(product(range(3), repeat=len(example.units)))
    states = {row.state for row in rows}
    if len(rows) != len(expected_states) or states != expected_states:
        errors.append(
            f"state coverage rows={len(rows)}, unique={len(states)}, "
            f"expected={len(expected_states)}"
        )
    if any(row.example_id != example.example_id for row in rows):
        errors.append("example id mismatch")
    if any(row.tokens < 0 for row in rows):
        errors.append("negative token count")
    for name in ("answer_em", "answer_f1", "fact_recall", "fidelity"):
        if any(not 0.0 <= float(getattr(row, name)) <= 1.0 for row in rows):
            errors.append(f"{name} outside [0,1]")
    if any(abs(row.fidelity - min(row.answer_f1, row.fact_recall)) > 1e-12 for row in rows):
        errors.append("fidelity/min metric mismatch")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate resumable V0 stage artifacts")
    parser.add_argument("kind", choices=("packets", "exact"))
    parser.add_argument("--examples", required=True)
    parser.add_argument("--artifact-dir", required=True)
    args = parser.parse_args()
    if args.kind == "packets":
        report = packet_status(args.examples, args.artifact_dir)
    else:
        report = exact_status(args.examples, args.artifact_dir)
    print(json.dumps(report, ensure_ascii=False))
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
