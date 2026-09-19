from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.data.schemas import read_jsonl
from src.reproducibility import sha256, write_metadata


ROOT = Path("results/v2_rank_then_cut")
OUTPUT = ROOT / "v17sel_b2b_lineage_clean_holdout"
CONFIG = Path("configs/v17sel_b2b_lineage_clean_pool_gate.json")


def fold(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % 5


def filter_jsonl(source: Path, output: Path, excluded: set[str], include: bool) -> dict:
    seen = set()
    count = 0
    with source.open(encoding="utf-8") as reader, output.open("w", encoding="utf-8") as writer:
        for line in reader:
            row = json.loads(line)
            query = row["example_id"]
            if query in seen:
                raise ValueError(f"duplicate ID in {source}: {query}")
            seen.add(query)
            if (query in excluded) == include:
                writer.write(line)
                count += 1
    return {"source": str(source), "source_sha256": sha256(source), "output": str(output), "output_sha256": sha256(output), "source_count": len(seen), "output_count": count}


def main() -> None:
    config = json.loads(CONFIG.read_text())
    if config["status"] != "FROZEN_APPROVED_TO_BUILD_AND_RUN":
        raise ValueError("lineage protocol is not frozen")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train_source = ROOT / "v13_internal_train2863_mask_value.jsonl"
    train_ids = [row["example_id"] for row in read_jsonl(train_source)]
    if len(train_ids) != 2863 or len(set(train_ids)) != 2863:
        raise ValueError("unexpected V13 source population")
    holdout = {query for query in train_ids if fold(query) == 1}
    if not 450 <= len(holdout) <= 700:
        raise ValueError(f"unexpected holdout count {len(holdout)}")
    data_dir = OUTPUT / "data"
    data_dir.mkdir(exist_ok=True)
    sources = {
        "v8_train": ROOT / "v8_train3163_history_supervision.jsonl",
        "v10_v12_train": ROOT / "v10_train3163_mask_value.jsonl",
        "v13_train": train_source,
    }
    manifest = {"protocol_sha256": sha256(CONFIG), "holdout_ids": sorted(holdout), "holdout_count": len(holdout), "files": {}}
    for name, source in sources.items():
        train_out = data_dir / f"{name}_without_holdout.jsonl"
        holdout_out = data_dir / f"{name}_holdout.jsonl"
        train_record = filter_jsonl(source, train_out, holdout, include=False)
        holdout_record = filter_jsonl(source, holdout_out, holdout, include=True)
        if holdout_record["output_count"] != len(holdout):
            raise ValueError(f"holdout incomplete in {name}")
        if train_record["output_count"] + holdout_record["output_count"] != train_record["source_count"]:
            raise ValueError(f"split count mismatch in {name}")
        manifest["files"][name] = {"train": train_record, "holdout": holdout_record}
    validation_paths = {
        "v8_development": ROOT / "v8_development300_history_supervision.jsonl",
        "v10_development": ROOT / "v10_development300_mask_value_full.jsonl",
        "v13_internal_validation": ROOT / "v13_internal_validation300_mask_value.jsonl",
    }
    manifest["validation"] = {}
    for name, path in validation_paths.items():
        ids = {row["example_id"] for row in read_jsonl(path)}
        if ids & holdout:
            raise ValueError(f"holdout enters checkpoint selection: {name}")
        manifest["validation"][name] = {"path": str(path), "sha256": sha256(path), "count": len(ids), "holdout_overlap": 0}
    write_metadata(OUTPUT / "manifest.json", manifest)
    print(json.dumps({"holdout_count": len(holdout), "train_counts": {name: value["train"]["output_count"] for name, value in manifest["files"].items()}, "validation_overlap": 0}, indent=2), flush=True)


if __name__ == "__main__":
    main()
