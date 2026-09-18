from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from src.data.schemas import read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata


def stratum(row: dict) -> tuple[str, str]:
    family = row["example_id"].split("__")[1]
    positives = int(row["class_population"].get("4", 0)) + int(
        row["class_population"].get("5", 0)
    )
    scarcity = "le4" if positives <= 4 else "5_10" if positives <= 10 else "gt10"
    return family, scarcity


def stable_key(example_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{example_id}".encode()).digest()


def split_rows(rows: list[dict], validation_size: int, seed: int):
    if not 0 < validation_size < len(rows):
        raise ValueError("validation size must be between zero and the population size")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    for values in groups.values():
        values.sort(key=lambda row: stable_key(row["example_id"], seed))
    quotas = {key: validation_size * len(values) / len(rows) for key, values in groups.items()}
    counts = {key: int(value) for key, value in quotas.items()}
    remaining = validation_size - sum(counts.values())
    order = sorted(groups, key=lambda key: (-(quotas[key] - counts[key]), key))
    for key in order[:remaining]:
        counts[key] += 1
    validation_ids = {
        row["example_id"]
        for key, values in groups.items()
        for row in values[: counts[key]]
    }
    train = [row for row in rows if row["example_id"] not in validation_ids]
    validation = [row for row in rows if row["example_id"] in validation_ids]
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a stratified internal V13 split")
    parser.add_argument("--source", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validation-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    rows = list(read_jsonl(args.source))
    train, validation = split_rows(rows, args.validation_size, args.seed)
    write_jsonl(args.train_output, train)
    write_jsonl(args.validation_output, validation)
    manifest = {
        "complete": True,
        "scientific_role": "train3163-only internal model-selection split",
        "seed": args.seed,
        "source": args.source,
        "source_sha256": sha256(args.source),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_sha256": sha256(args.train_output),
        "validation_sha256": sha256(args.validation_output),
        "train_strata": {str(key): sum(stratum(row) == key for row in train) for key in sorted({stratum(row) for row in rows})},
        "validation_strata": {str(key): sum(stratum(row) == key for row in validation) for key in sorted({stratum(row) for row in rows})},
        "locked_roles_used": False,
    }
    write_metadata(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
