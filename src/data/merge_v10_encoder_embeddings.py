from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import torch

from src.data.schemas import read_jsonl
from src.reproducibility import sha256, write_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge V10 embedding-cache shards")
    parser.add_argument("--data", required=True)
    parser.add_argument("--part", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    expected = [row["example_id"] for row in read_jsonl(args.data)]
    records = {}
    part_hashes = {}
    for name in args.part:
        cache = torch.load(name, map_location="cpu", weights_only=True)
        part_hashes[name] = sha256(name)
        for index, example_id in enumerate(cache["example_ids"]):
            if example_id in records:
                raise ValueError(f"duplicate cached example {example_id}")
            records[example_id] = (cache["packets"][index], cache["questions"][index])
    if set(records) != set(expected):
        raise ValueError("cache shards do not exactly cover the input data")
    payload = {
        "example_ids": expected,
        "packets": torch.stack([records[value][0] for value in expected]),
        "questions": torch.stack([records[value][1] for value in expected]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    write_metadata(args.manifest, {
        "complete": True,
        "data": args.data,
        "data_sha256": sha256(args.data),
        "examples": len(expected),
        "parts": part_hashes,
        "shape_packets": list(payload["packets"].shape),
        "shape_questions": list(payload["questions"].shape),
        "dtype": str(payload["packets"].dtype),
        "output": str(output),
        "output_sha256": sha256(output),
    })


if __name__ == "__main__":
    main()
