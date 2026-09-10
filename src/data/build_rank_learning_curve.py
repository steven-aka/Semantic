from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.schemas import read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata


SIZES = (500, 1000, 2000)


def build_nested_learning_curve(
    rows: list[dict[str, object]],
) -> dict[int, list[dict[str, object]]]:
    if len(rows) != SIZES[-1]:
        raise ValueError(f"frozen V2 ranking train set must contain {SIZES[-1]} rows")
    ids = [str(row["example_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("ranking train set contains duplicate example ids")
    if any(not row.get("pairwise_preferences") for row in rows):
        raise ValueError("every ranking row must contain identifiable preferences")
    return {size: rows[:size] for size in SIZES}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen nested V2 ranking learning curve")
    parser.add_argument("--train2000-oracle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    source = list(read_jsonl(args.train2000_oracle))
    subsets = build_nested_learning_curve(source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "protocol": "v2_rank_then_cut_nested_learning_curve",
        "source": args.train2000_oracle,
        "source_sha256": sha256(args.train2000_oracle),
        "sizes": {},
        "nested": True,
    }
    prior_ids: set[str] = set()
    for size, rows in subsets.items():
        path = output_dir / f"train{size}_rank_oracle.jsonl"
        write_jsonl(path, rows)
        ids = {str(row["example_id"]) for row in rows}
        if prior_ids and not prior_ids < ids:
            raise RuntimeError("learning-curve prefix nesting invariant failed")
        prior_ids = ids
        manifest["sizes"][str(size)] = {
            "path": str(path),
            "rows": len(rows),
            "sha256": sha256(path),
        }
    write_metadata(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
