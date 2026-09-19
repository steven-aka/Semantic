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


def filter_jsonl(source: Path, output: Path, selected: set[str]) -> dict:
    seen = set()
    count = 0
    with source.open(encoding="utf-8") as reader, output.open("w", encoding="utf-8") as writer:
        for line in reader:
            row = json.loads(line)
            query = row["example_id"]
            if query in seen:
                raise ValueError(f"duplicate ID in {source}: {query}")
            seen.add(query)
            if query in selected:
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
    inner_validation = {query for query in train_ids if fold(query) == 2}
    if not 450 <= len(holdout) <= 700:
        raise ValueError(f"unexpected holdout count {len(holdout)}")
    if not 450 <= len(inner_validation) <= 700 or holdout & inner_validation:
        raise ValueError(f"invalid inner validation count {len(inner_validation)}")
    data_dir = OUTPUT / "data"
    data_dir.mkdir(exist_ok=True)
    sources = {
        "v8_train": ROOT / "v8_train3163_history_supervision.jsonl",
        "v10_v12_train": ROOT / "v10_train3163_mask_value.jsonl",
        "v13_train": train_source,
    }
    manifest = {"protocol_sha256": sha256(CONFIG), "holdout_ids": sorted(holdout), "holdout_count": len(holdout), "inner_validation_ids": sorted(inner_validation), "inner_validation_count": len(inner_validation), "files": {}}
    for name, source in sources.items():
        source_ids = {row["example_id"] for row in read_jsonl(source)}
        train_out = data_dir / f"{name}_train_clean.jsonl"
        holdout_out = data_dir / f"{name}_holdout.jsonl"
        inner_out = data_dir / f"{name}_inner_validation.jsonl"
        train_record = filter_jsonl(source, train_out, source_ids - holdout - inner_validation)
        holdout_record = filter_jsonl(source, holdout_out, holdout)
        inner_record = filter_jsonl(source, inner_out, inner_validation)
        if holdout_record["output_count"] != len(holdout):
            raise ValueError(f"holdout incomplete in {name}")
        if inner_record["output_count"] != len(inner_validation):
            raise ValueError(f"inner validation incomplete in {name}")
        if train_record["output_count"] + holdout_record["output_count"] + inner_record["output_count"] != train_record["source_count"]:
            raise ValueError(f"split count mismatch in {name}")
        manifest["files"][name] = {"train": train_record, "holdout": holdout_record, "inner_validation": inner_record}
    for name in ("v8_train", "v10_v12_train", "v13_train"):
        actual_train = {row["example_id"] for row in read_jsonl(manifest["files"][name]["train"]["output"])}
        actual_inner = {row["example_id"] for row in read_jsonl(manifest["files"][name]["inner_validation"]["output"])}
        if actual_train & (holdout | inner_validation) or actual_inner != inner_validation:
            raise ValueError(f"lineage role overlap for {name}")
    manifest["sealed_existing_roles"] = {
        "v8_development": str(ROOT / "v8_development300_history_supervision.jsonl"),
        "v10_development": str(ROOT / "v10_development300_mask_value_full.jsonl"),
        "v13_internal_validation": str(ROOT / "v13_internal_validation300_mask_value.jsonl"),
    }
    write_metadata(OUTPUT / "manifest.json", manifest)
    print(json.dumps({"holdout_count": len(holdout), "inner_validation_count": len(inner_validation), "train_counts": {name: value["train"]["output_count"] for name, value in manifest["files"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
