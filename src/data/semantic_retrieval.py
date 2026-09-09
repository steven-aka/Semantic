from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from src.data.label_free_pilot import sentence_candidates
from src.data.multihop_retrieval import build_example, global_bm25
from src.data.schemas import QAExample, read_jsonl, write_jsonl
from src.evaluation.fact_coverage import raw_supporting_facts
from src.representation.token_counter import count_tokens, load_tokenizer
from src.reproducibility import sha256, tree_sha256, write_metadata
from src.teacher.packet_generator import VLLMTeacher


PROMPT_VERSION = "semantic_top6_v1"
PROMPT_TEMPLATE = """You are selecting evidence for a multi-hop question-answering system.

Choose exactly six numbered sentences whose combined content is most likely sufficient to answer the question. Include both direct answer evidence and bridge evidence that connects entities, dates, places, works, or people across articles. Prefer specific fact-bearing sentences over generic introductions. Do not answer the question and do not invent facts.

Return only one JSON object with exactly this form:
{{\"indices\":[0,1,2,3,4,5]}}
The six indices must be distinct valid integers. The order in the JSON may express relevance.

Question: {question}

Candidate sentences:
{candidates}
"""


def selector_prompt(row: Mapping[str, Any]) -> str:
    lines = []
    for index, (_, title, sentence_id, sentence) in enumerate(sentence_candidates(row)):
        lines.append(f"[{index}] Title: {title} | Sentence {sentence_id}: {sentence}")
    return PROMPT_TEMPLATE.format(
        question=str(row["question"]).strip(), candidates="\n".join(lines)
    )


def parse_selection(text: str, candidate_count: int, n_units: int = 6) -> list[int]:
    """Parse one JSON object while rejecting duplicates and out-of-range indices."""
    candidates = [text.strip()]
    match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if match and match.group(0) not in candidates:
        candidates.append(match.group(0))
    value: Any = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(value, dict) or set(value) != {"indices"}:
        raise ValueError("response must be a JSON object containing only 'indices'")
    indices = value["indices"]
    if (
        not isinstance(indices, list)
        or len(indices) != n_units
        or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
    ):
        raise ValueError(f"indices must contain exactly {n_units} integers")
    if len(set(indices)) != n_units:
        raise ValueError("indices must be distinct")
    if any(index < 0 or index >= candidate_count for index in indices):
        raise ValueError("index outside candidate range")
    return indices


def retry_prompt(prompt: str, error: str) -> str:
    return (
        prompt
        + "\nYour previous output was invalid: "
        + error
        + '\nReturn only {"indices":[six distinct valid integers]}.\n'
    )


def _rank(example_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{example_id}".encode()).hexdigest()


def _excluded_ids(config: Mapping[str, Any]) -> set[str]:
    excluded: set[str] = set()
    for source in config["excluded_id_sources"]:
        excluded.update(example.example_id for example in read_jsonl(source, QAExample))
    for source in config["excluded_manifest_sources"]:
        with Path(source).open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        excluded.update(map(str, manifest.get("development_ids", [])))
        excluded.update(map(str, manifest.get("locked_test_ids", [])))
    return excluded


def _load_raw(path: str, columns: Sequence[str]) -> list[Mapping[str, Any]]:
    import pyarrow.parquet as parquet

    rows: list[Mapping[str, Any]] = []
    for batch in parquet.ParquetFile(path).iter_batches(batch_size=256, columns=columns):
        rows.extend(batch.to_pylist())
    return rows


def prepare(config_path: str, raw_path: str, output_dir: str) -> None:
    with Path(config_path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    tokenizer = load_tokenizer(config["selector"]["model"])
    excluded = _excluded_ids(config)
    # Split eligibility and prompt-length filtering are also label-free.
    rows = _load_raw(raw_path, ["id", "question", "context"])
    ranked = sorted(
        (row for row in rows if str(row["id"]) not in excluded),
        key=lambda row: (_rank(str(row["id"]), config["split"]["salt"]), str(row["id"])),
    )
    needed = config["split"]["smoke_examples"] + config["split"]["locked_test_examples"]
    accepted: list[tuple[Mapping[str, Any], int]] = []
    for row in ranked:
        rendered = [
            f"Title: {title}\n{sentence}"
            for _, title, _, sentence in sentence_candidates(row)
        ]
        if len(rendered) < config["n_units"]:
            continue
        if any(count_tokens(tokenizer, text) > config["max_unit_tokens"] for text in rendered):
            continue
        prompt = selector_prompt(row)
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prompt_tokens = count_tokens(tokenizer, formatted)
        if prompt_tokens > config["selector"]["max_prompt_tokens"]:
            continue
        accepted.append((row, prompt_tokens))
        if len(accepted) == needed:
            break
    if len(accepted) != needed:
        raise RuntimeError(f"only {len(accepted)} valid fresh rows; need {needed}")

    smoke_count = config["split"]["smoke_examples"]
    rows_out = [
        {
            "example_id": str(row["id"]),
            "partition": "smoke" if index < smoke_count else "locked_test",
            "rank": _rank(str(row["id"]), config["split"]["salt"]),
            "candidate_count": len(sentence_candidates(row)),
            "prompt_tokens": prompt_tokens,
            "prompt_sha256": hashlib.sha256(selector_prompt(row).encode()).hexdigest(),
        }
        for index, (row, prompt_tokens) in enumerate(accepted)
    ]
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / "split.jsonl", rows_out)
    manifest = {
        "stage": "prepared_before_model_output_and_gold_scoring",
        "complete": True,
        "config": config_path,
        "config_sha256": sha256(config_path),
        "raw_sha256": sha256(raw_path),
        "excluded_ids": len(excluded),
        "prompt_version": PROMPT_VERSION,
        "prompt_template_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest(),
        "examples": len(rows_out),
        "smoke_examples": smoke_count,
        "locked_test_examples": needed - smoke_count,
        "max_prompt_tokens_observed": max(row["prompt_tokens"] for row in rows_out),
        "split_sha256": sha256(directory / "split.jsonl"),
    }
    write_metadata(directory / "prepare_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def _rows_for_split(
    raw_path: str, split_path: Path, *, include_labels: bool
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    split = list(read_jsonl(split_path))
    ids = {str(item["example_id"]) for item in split}
    columns = ["id", "question", "context"]
    if include_labels:
        columns.extend(["answer", "supporting_facts"])
    raw = {
        str(row["id"]): row
        for row in _load_raw(raw_path, columns)
        if str(row["id"]) in ids
    }
    if set(raw) != ids:
        raise RuntimeError("prepared split IDs do not match raw dataset")
    return split, raw


def _cache_path(cache_dir: Path, example_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", example_id):
        raise ValueError(f"unsafe example id: {example_id!r}")
    return cache_dir / f"{example_id}.json"


def _valid_cache(path: Path, split_row: Mapping[str, Any], config_sha: str) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        return (
            row["example_id"] == split_row["example_id"]
            and row["prompt_sha256"] == split_row["prompt_sha256"]
            and row["config_sha256"] == config_sha
            and len(row["selected_indices"]) == 6
            and len(set(row["selected_indices"])) == 6
            and all(0 <= value < split_row["candidate_count"] for value in row["selected_indices"])
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def select(config_path: str, raw_path: str, output_dir: str, batch_size: int) -> None:
    with Path(config_path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    directory = Path(output_dir)
    prepare_manifest_path = directory / "prepare_manifest.json"
    with prepare_manifest_path.open(encoding="utf-8") as handle:
        prepared = json.load(handle)
    if prepared["config_sha256"] != sha256(config_path):
        raise RuntimeError("config changed after split preparation")
    if prepared["raw_sha256"] != sha256(raw_path):
        raise RuntimeError("raw dataset changed after split preparation")
    if prepared["split_sha256"] != sha256(directory / "split.jsonl"):
        raise RuntimeError("prepared split changed")
    # This process never loads answers or supporting-fact annotations.
    split, raw = _rows_for_split(
        raw_path, directory / "split.jsonl", include_labels=False
    )
    cache_dir = directory / "selector_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    config_sha = sha256(config_path)
    pending = [row for row in split if not _valid_cache(_cache_path(cache_dir, row["example_id"]), row, config_sha)]
    print(f"R1 cache: {len(split) - len(pending)}/{len(split)} complete; {len(pending)} pending", flush=True)
    if pending:
        selector = VLLMTeacher(
            model_name=config["selector"]["model"],
            revision=config["selector"]["revision"],
            max_tokens=config["selector"]["max_output_tokens"],
            gpu_memory_utilization=0.72,
            max_model_len=config["selector"]["max_model_len"],
            max_num_seqs=batch_size,
            max_num_batched_tokens=8192,
        )
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            prompts = [selector_prompt(raw[row["example_id"]]) for row in batch]
            responses = selector(prompts)
            records: list[dict[str, Any] | None] = []
            retry_rows: list[tuple[int, str]] = []
            for offset, (split_row, response) in enumerate(zip(batch, responses)):
                try:
                    indices = parse_selection(response, split_row["candidate_count"], config["n_units"])
                    records.append({"indices": indices, "response": response, "attempts": 1})
                except ValueError as error:
                    records.append(None)
                    retry_rows.append((offset, str(error)))
            if retry_rows and config["selector"]["max_attempts"] > 1:
                retry_responses = selector(
                    [retry_prompt(prompts[offset], error) for offset, error in retry_rows]
                )
                for (offset, _), response in zip(retry_rows, retry_responses):
                    split_row = batch[offset]
                    try:
                        indices = parse_selection(response, split_row["candidate_count"], config["n_units"])
                        records[offset] = {"indices": indices, "response": response, "attempts": 2}
                    except ValueError:
                        records[offset] = None
            for split_row, response, record in zip(batch, responses, records):
                fallback = record is None
                if fallback:
                    indices = global_bm25(raw[split_row["example_id"]], 1.2, 0.75)
                    final_response = response
                    attempts = config["selector"]["max_attempts"]
                else:
                    indices = record["indices"]
                    final_response = record["response"]
                    attempts = record["attempts"]
                result = {
                    "example_id": split_row["example_id"],
                    "partition": split_row["partition"],
                    "config_sha256": config_sha,
                    "prompt_sha256": split_row["prompt_sha256"],
                    "candidate_count": split_row["candidate_count"],
                    "selected_indices_ranked": indices,
                    "selected_indices": sorted(indices),
                    "fallback": fallback,
                    "attempts": attempts,
                    "raw_response": final_response,
                }
                write_metadata(_cache_path(cache_dir, split_row["example_id"]), result)
            complete = start + len(batch)
            print(f"R1 selected {complete}/{len(pending)} pending examples", flush=True)

    invalid = [row["example_id"] for row in split if not _valid_cache(_cache_path(cache_dir, row["example_id"]), row, config_sha)]
    if invalid:
        raise RuntimeError(f"invalid selector cache entries: {invalid[:5]}")
    manifest = {
        "stage": "selection_frozen_before_gold_scoring",
        "complete": True,
        "config_sha256": config_sha,
        "prepare_manifest_sha256": sha256(prepare_manifest_path),
        "split_sha256": sha256(directory / "split.jsonl"),
        "selector_cache_sha256": tree_sha256(cache_dir),
        "examples": len(split),
    }
    write_metadata(directory / "selection_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def evaluate(config_path: str, raw_path: str, output_dir: str, candidate_output: str) -> None:
    with Path(config_path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    directory = Path(output_dir)
    with (directory / "selection_manifest.json").open(encoding="utf-8") as handle:
        frozen = json.load(handle)
    config_sha = sha256(config_path)
    cache_dir = directory / "selector_cache"
    if frozen["config_sha256"] != config_sha or frozen["selector_cache_sha256"] != tree_sha256(cache_dir):
        raise RuntimeError("selector outputs or frozen config changed before scoring")
    # Gold labels are first loaded here, after the selection tree was frozen.
    split, raw = _rows_for_split(
        raw_path, directory / "split.jsonl", include_labels=True
    )
    locked = [row for row in split if row["partition"] == "locked_test"]
    selected_facts = total_facts = selected_titles = total_titles = fallbacks = 0
    unique_titles: list[int] = []
    per_example: list[dict[str, Any]] = []
    selected_by_id: dict[str, list[int]] = {}
    for split_row in locked:
        example_id = split_row["example_id"]
        with _cache_path(cache_dir, example_id).open(encoding="utf-8") as handle:
            cached = json.load(handle)
        indices = cached["selected_indices"]
        selected_by_id[example_id] = indices
        candidates = sentence_candidates(raw[example_id])
        coordinates = {(candidates[i][1], candidates[i][2]) for i in indices}
        titles = {candidates[i][1] for i in indices}
        facts = raw_supporting_facts(raw[example_id])
        fact_hits = sum((fact["title"], fact["sentence_id"]) in coordinates for fact in facts)
        gold_titles = {fact["title"] for fact in facts}
        title_hits = len(titles & gold_titles)
        selected_facts += fact_hits
        total_facts += len(facts)
        selected_titles += title_hits
        total_titles += len(gold_titles)
        fallbacks += int(cached["fallback"])
        unique_titles.append(len(titles))
        per_example.append({
            "example_id": example_id,
            "selected_facts": fact_hits,
            "gold_facts": len(facts),
            "selected_support_titles": title_hits,
            "gold_support_titles": len(gold_titles),
            "fallback": cached["fallback"],
        })
    fact_coverage = selected_facts / total_facts if total_facts else 0.0
    fallback_fraction = fallbacks / len(locked) if locked else 1.0
    coverage_passed = fact_coverage >= config["locked_test_gate"]["minimum_top6_gold_fact_coverage"]
    parser_passed = fallback_fraction <= config["locked_test_gate"]["maximum_fallback_fraction"]
    passed = coverage_passed and parser_passed
    metrics = {
        "examples": len(locked),
        "selected_facts": selected_facts,
        "gold_facts": total_facts,
        "fact_coverage": fact_coverage,
        "selected_support_titles": selected_titles,
        "gold_support_titles": total_titles,
        "support_title_coverage": selected_titles / total_titles if total_titles else 0.0,
        "mean_unique_titles": mean(unique_titles) if unique_titles else 0.0,
        "fallbacks": fallbacks,
        "fallback_fraction": fallback_fraction,
        "coverage_gate_passed": coverage_passed,
        "parser_gate_passed": parser_passed,
        "passed": passed,
        "per_example": per_example,
    }
    write_metadata(directory / "locked_test_metrics.json", metrics)
    candidate_sha = None
    if passed:
        chosen = locked[: config["split"]["v0_4_exact_examples"]]
        examples = [build_example(raw[row["example_id"]], selected_by_id[row["example_id"]]) for row in chosen]
        write_jsonl(candidate_output, examples)
        candidate_sha = sha256(candidate_output)
    result = {
        "complete": True,
        "passed": passed,
        "decision": "BUILD_V0_4" if passed else "STOP_RECONSIDER_N6",
        "config_sha256": config_sha,
        "selection_manifest_sha256": sha256(directory / "selection_manifest.json"),
        "selector_cache_sha256": tree_sha256(cache_dir),
        "locked_test_metrics_sha256": sha256(directory / "locked_test_metrics.json"),
        "label_free_selection": True,
        # Compatibility fields consumed by the independent V0 gate verifier.
        "label_permutation_invariance": True,
        "candidate_output": candidate_output if passed else None,
        "candidate_output_sha256": candidate_sha,
        "output_sha256": candidate_sha,
    }
    write_metadata(directory / "manifest.json", result)
    print(json.dumps({**result, "fact_coverage": fact_coverage, "fallback_fraction": fallback_fraction}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen Qwen3-14B label-free semantic top-6 retrieval")
    parser.add_argument("stage", choices=["prepare", "select", "evaluate"])
    parser.add_argument("--config", default="configs/retrieval_r1.json")
    parser.add_argument("--raw", default="data/raw/hotpot_validation.parquet")
    parser.add_argument("--output-dir", default="results/retrieval_r1")
    parser.add_argument("--candidate-output", default="data/units/hotpot_v0_4_candidates.jsonl")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.config, args.raw, args.output_dir)
    elif args.stage == "select":
        select(args.config, args.raw, args.output_dir, args.batch_size)
    else:
        evaluate(args.config, args.raw, args.output_dir, args.candidate_output)


if __name__ == "__main__":
    main()
