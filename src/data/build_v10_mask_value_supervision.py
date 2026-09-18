from __future__ import annotations

import argparse
import hashlib
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.search.exact_frontier import C_GRID


def stable_seed(example_id: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{example_id}".encode()).digest()[:8], "big")


def build_example(
    source: dict[str, Any], exact_dir: str, per_class: int | None, seed: int
) -> dict[str, Any]:
    exact = list(
        read_jsonl(Path(exact_dir) / f"{source['example_id']}.jsonl", ExactSearchResult)
    )
    if not exact:
        raise ValueError(f"empty exact lattice for {source['example_id']}")
    full_mask = (1 << len(exact[0].state)) - 1
    levels = [float(value) for value in C_GRID]
    attainable_levels = [float(value) for value in source["active_levels"]]
    groups: dict[int, list[tuple[int, int]]] = {value: [] for value in range(len(levels) + 1)}
    for row in exact:
        attained = sum(float(row.fidelity) + 1e-12 >= level for level in levels)
        groups[attained].append((state_to_mask(row.state), int(row.tokens)))
    rng = random.Random(stable_seed(source["example_id"], seed))
    selected = []
    for attained, values in groups.items():
        if not values:
            continue
        if per_class is None or len(values) <= per_class:
            chosen = [(mask, tokens, 1.0) for mask, tokens in values]
        else:
            # Empty and full masks exercise special zero-pool branches in the
            # value head and anchor the two ends of every decoded lattice.
            required = [value for value in values if value[0] in (0, full_mask)]
            remainder = [value for value in values if value[0] not in (0, full_mask)]
            sampled = rng.sample(remainder, per_class - len(required))
            probability = (per_class - len(required)) / len(remainder)
            chosen = [(mask, tokens, 1.0) for mask, tokens in required]
            chosen.extend((mask, tokens, probability) for mask, tokens in sampled)
        selected.extend((mask, tokens, attained, probability) for mask, tokens, probability in chosen)
    selected.sort()
    return {
        "example_id": source["example_id"],
        "question": source["question"],
        "packet_texts": source["packet_texts"],
        "packet_tokens": source["packet_tokens"],
        "active_levels": levels,
        "attainable_levels": attainable_levels,
        "mask_values": [
            {
                "mask": mask,
                "tokens": tokens,
                "attained_levels": attained,
                "sampling_probability": probability,
            }
            for mask, tokens, attained, probability in selected
        ],
        "class_population": {str(key): len(value) for key, value in groups.items()},
    }


def _worker(payload: tuple[dict[str, Any], str, int | None, int]):
    return build_example(*payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stratified mask-value supervision")
    parser.add_argument("--source", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--per-class", type=int, help="omit to retain the full lattice")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    forbidden = ("calibration300", "final_test300", "test300")
    if any(token in args.source.lower() for token in forbidden):
        raise ValueError("locked calibration/final-test roles must not enter V10")
    sources = list(read_jsonl(args.source))
    payloads = ((row, args.exact_dir, args.per_class, args.seed) for row in sources)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(pool.map(_worker, payloads, chunksize=4), 1):
            rows.append(row)
            if index % 100 == 0 or index == len(sources):
                print(json.dumps({"processed": index, "total": len(sources)}), flush=True)
    write_jsonl(args.output, rows)
    class_samples: dict[str, int] = {}
    class_population: dict[str, int] = {}
    for row in rows:
        for item in row["mask_values"]:
            key = str(item["attained_levels"])
            class_samples[key] = class_samples.get(key, 0) + 1
        for key, value in row["class_population"].items():
            class_population[key] = class_population.get(key, 0) + int(value)
    manifest = {
        "complete": True,
        "examples": len(rows),
        "states": sum(len(row["mask_values"]) for row in rows),
        "per_class_limit": args.per_class,
        "class_samples": class_samples,
        "class_population": class_population,
        "source": args.source,
        "source_sha256": sha256(args.source),
        "output": args.output,
        "output_sha256": sha256(args.output),
        "seed": args.seed,
        "locked_roles_used": False,
    }
    write_metadata(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
