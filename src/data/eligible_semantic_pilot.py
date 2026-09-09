from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from src.data.multihop_retrieval import build_example
from src.data.schemas import QAExample, read_jsonl, write_jsonl
from src.data.semantic_retrieval import selector_prompt
from src.evaluation.full_context import cached_full_context_valid
from src.representation.token_counter import load_tokenizer
from src.reproducibility import sha256, tree_sha256, write_metadata


def _load_raw(path: str, wanted: set[str]) -> dict[str, Mapping[str, Any]]:
    import pyarrow.parquet as parquet

    output: dict[str, Mapping[str, Any]] = {}
    columns = ["id", "question", "answer", "context", "supporting_facts"]
    for batch in parquet.ParquetFile(path).iter_batches(batch_size=256, columns=columns):
        for row in batch.to_pylist():
            if str(row["id"]) in wanted:
                output[str(row["id"])] = row
    return output


def prepare(config_path: str, raw_path: str, output_dir: str) -> None:
    with Path(config_path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    selection = config["selection"]
    with Path(selection["selection_manifest"]).open(encoding="utf-8") as handle:
        frozen = json.load(handle)
    cache_dir = Path(selection["selector_cache"])
    if frozen.get("selector_cache_sha256") != tree_sha256(cache_dir):
        raise RuntimeError("Retrieval R1 selector cache changed after freezing")
    split = list(read_jsonl(selection["split"]))
    locked = [row for row in split if row.get("partition") == "locked_test"]
    offset = int(selection["locked_offset"])
    pool_rows = locked[offset : offset + config["sample"]["pool_examples"]]
    if len(pool_rows) != config["sample"]["pool_examples"]:
        raise RuntimeError("frozen Retrieval R1 pool is shorter than preregistered")
    ids = {str(row["example_id"]) for row in pool_rows}
    raw = _load_raw(raw_path, ids)
    if set(raw) != ids:
        raise RuntimeError("raw rows do not match frozen V0.5 pool")
    examples: list[QAExample] = []
    for split_row in pool_rows:
        example_id = str(split_row["example_id"])
        with (cache_dir / f"{example_id}.json").open(encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached["prompt_sha256"] != split_row["prompt_sha256"]:
            raise RuntimeError(f"{example_id}: prompt/cache mismatch")
        examples.append(build_example(raw[example_id], cached["selected_indices"]))
    pool_path = Path(selection["eligibility_pool"])
    write_jsonl(pool_path, examples)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    write_metadata(directory / "pool_manifest.json", {
        "complete": True,
        "stage": "v0_5_pool_frozen_before_baseline",
        "config_sha256": sha256(config_path),
        "selection_manifest_sha256": sha256(selection["selection_manifest"]),
        "selector_cache_sha256": tree_sha256(cache_dir),
        "pool_examples": len(examples),
        "pool_ids": [example.example_id for example in examples],
        "pool_sha256": sha256(pool_path),
    })
    print(f"[v0.5 prepare] froze {len(examples)} answerability-pool examples")


def finalize(config_path: str, raw_path: str, output_dir: str, candidate_output: str) -> None:
    with Path(config_path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    selection = config["selection"]
    directory = Path(output_dir)
    with (directory / "pool_manifest.json").open(encoding="utf-8") as handle:
        pool_manifest = json.load(handle)
    if pool_manifest["config_sha256"] != sha256(config_path):
        raise RuntimeError("V0.5 config changed after pool preparation")
    if pool_manifest["pool_sha256"] != sha256(selection["eligibility_pool"]):
        raise RuntimeError("V0.5 pool changed after preparation")
    pool = list(read_jsonl(selection["eligibility_pool"], QAExample))
    baseline = list(read_jsonl(selection["eligibility_baseline"]))
    threshold = float(config["sample"]["eligibility_f1"])
    eligible = [
        example for example, row in zip(pool, baseline)
        if float(row["f1"]) >= threshold
    ]
    needed = config["sample"]["n_examples"]
    if len(eligible) < needed:
        raise RuntimeError(f"only {len(eligible)} answerable examples; need {needed}")
    chosen = eligible[:needed]
    tokenizer = load_tokenizer(config["models"]["target_and_judge"])
    if not cached_full_context_valid(pool, baseline, eligible, tokenizer):
        raise RuntimeError("pool full-context baseline or eligibility list is invalid")

    selected_ids = {example.example_id for example in chosen}
    selected_baseline = [row for row in baseline if str(row["example_id"]) in selected_ids]
    if [row["example_id"] for row in selected_baseline] != [example.example_id for example in chosen]:
        raise RuntimeError("selected baseline order mismatch")
    write_jsonl(candidate_output, chosen)
    write_jsonl(directory / "full_context.jsonl", selected_baseline)

    raw = _load_raw(raw_path, selected_ids)
    label_invariant = True
    for example in chosen:
        row = raw[example.example_id]
        perturbed = {
            **row,
            "answer": "LABEL_ERASED",
            "supporting_facts": {"title": [], "sent_id": []},
        }
        label_invariant &= selector_prompt(row) == selector_prompt(perturbed)
    manifest = {
        "complete": True,
        "stage": "v0_5_candidates_frozen_before_packets_and_exact_search",
        "config_sha256": sha256(config_path),
        "pool_manifest_sha256": sha256(directory / "pool_manifest.json"),
        "pool_baseline_sha256": sha256(selection["eligibility_baseline"]),
        "pool_examples": len(pool),
        "eligible_examples": len(eligible),
        "selected_examples": len(chosen),
        "selected_ids": [example.example_id for example in chosen],
        "eligibility_f1": threshold,
        "label_permutation_invariance": bool(label_invariant),
        "output_sha256": sha256(candidate_output),
        "selected_baseline_sha256": sha256(directory / "full_context.jsonl"),
    }
    write_metadata(directory / "data_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the plan-aligned answerable V0.5 population")
    parser.add_argument("stage", choices=["prepare", "finalize"])
    parser.add_argument("--config", default="configs/v0_5_gate.json")
    parser.add_argument("--raw", default="data/raw/hotpot_validation.parquet")
    parser.add_argument("--output-dir", default="results/v0_5")
    parser.add_argument("--candidate-output", default="data/units/hotpot_v0_5_candidates.jsonl")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.config, args.raw, args.output_dir)
    else:
        finalize(args.config, args.raw, args.output_dir, args.candidate_output)


if __name__ == "__main__":
    main()
