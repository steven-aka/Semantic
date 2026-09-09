from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from src.data.label_free_pilot import build_label_free_example, unit_selection_signature
from src.data.multihop_retrieval import build_example
from src.data.semantic_retrieval import selector_prompt
from src.data.schemas import ExactSearchResult, QAExample, read_jsonl
from src.evaluation.artifact_status import packet_status, validate_exact_rows
from src.evaluation.fact_coverage import (
    _load_selected_raw,
    raw_supporting_facts,
    validate_fact_coverage,
)
from src.evaluation.qa_metrics import exact_match, token_f1
from src.evaluation.rescore_fact_fidelity import rescore_results
from src.representation.packet_store import PacketStore
from src.representation.state_builder import build_representation
from src.representation.token_counter import count_tokens, load_tokenizer
from src.reproducibility import sha256, tree_sha256, write_metadata
from src.search.best_nested_chain import best_nested_chain, is_nested, structural_gap_rows
from src.search.exact_frontier import C_GRID, attainable_levels, independent_frontier
from src.target.answer_parser import parse_answer


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _full_raw(path: str | Path, wanted: set[str]) -> dict[str, Mapping[str, Any]]:
    import pyarrow.parquet as parquet

    output = {}
    for batch in parquet.ParquetFile(path).iter_batches(
        batch_size=256,
        columns=["id", "question", "answer", "context", "supporting_facts"],
    ):
        for row in batch.to_pylist():
            if row["id"] in wanted:
                output[row["id"]] = row
    return output


def optimal_states(
    results: Sequence[ExactSearchResult], level: float
) -> tuple[int | None, list[tuple[int, ...]]]:
    feasible = [row for row in results if row.fidelity >= level]
    if not feasible:
        return None, []
    minimum = min(row.tokens for row in feasible)
    return minimum, [row.state for row in feasible if row.tokens == minimum]


def _csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_form(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {key: "" if value is None else str(value) for key, value in row.items()}
        for row in rows
    ]


def _keyed(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, str]]:
    normalized = _csv_form(rows)
    return {
        (row["example_id"], row["fidelity_level"]): row
        for row in normalized
    }


def evaluate_gate(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    candidates = list(read_jsonl(args.examples, QAExample))
    tokenizer = load_tokenizer(args.tokenizer)
    with Path(args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    thresholds = config["hard_gates"]
    with Path(args.manifest).open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("output_sha256") != sha256(args.examples):
        errors.append("data manifest candidate hash mismatch")
    if not manifest.get("label_permutation_invariance"):
        errors.append("label permutation invariant was not certified")
    if len(candidates) != config["sample"]["n_examples"]:
        errors.append("candidate count differs from preregistration")
    if any(len(example.units) != config["sample"]["n_units"] for example in candidates):
        errors.append("not every candidate has exactly six units")
    if any(unit.supporting for example in candidates for unit in example.units):
        errors.append("label-free candidate contains a supporting=True construction label")

    raw_full = _full_raw(args.raw, {example.example_id for example in candidates})
    if len(raw_full) != len(candidates):
        errors.append("raw rows missing for candidates")
    else:
        if config["selection"]["policy"] == "frozen_qwen3_14b_semantic_top6":
            selection_manifest_path = Path(config["selection"]["selection_manifest"])
            selector_cache = Path(config["selection"]["selector_cache"])
            split_path = Path(config["selection"]["split"])
            try:
                with selection_manifest_path.open(encoding="utf-8") as handle:
                    selection_manifest = json.load(handle)
                if selection_manifest.get("selector_cache_sha256") != tree_sha256(selector_cache):
                    errors.append("semantic selector cache changed after freezing")
                split_rows = list(read_jsonl(split_path))
                locked_ids = [
                    str(row["example_id"])
                    for row in split_rows
                    if row.get("partition") == "locked_test"
                ]
                offset = int(config["selection"].get("locked_offset", 0))
                if "eligibility_baseline" in config["selection"]:
                    pool_size = int(config["sample"]["pool_examples"])
                    pool_ids = locked_ids[offset : offset + pool_size]
                    baseline_pool = list(read_jsonl(config["selection"]["eligibility_baseline"]))
                    baseline_by_id = {str(row["example_id"]): row for row in baseline_pool}
                    expected_ids = [
                        example_id for example_id in pool_ids
                        if float(baseline_by_id[example_id]["f1"])
                        >= float(config["sample"]["eligibility_f1"])
                    ][: config["sample"]["n_examples"]]
                else:
                    expected_ids = locked_ids[offset : offset + config["sample"]["n_examples"]]
                if expected_ids != [example.example_id for example in candidates]:
                    errors.append("semantic candidates are not the frozen expected locked-test examples")
                for example in candidates:
                    with (selector_cache / f"{example.example_id}.json").open(encoding="utf-8") as handle:
                        cached = json.load(handle)
                    rebuilt = build_example(raw_full[example.example_id], cached["selected_indices"])
                    if unit_selection_signature(rebuilt) != unit_selection_signature(example):
                        errors.append(f"{example.example_id}: frozen semantic selection is not reproducible")
                    perturbed = {
                        **raw_full[example.example_id],
                        "answer": "LABEL_ERASED",
                        "supporting_facts": {"title": [], "sent_id": []},
                    }
                    if selector_prompt(perturbed) != selector_prompt(raw_full[example.example_id]):
                        errors.append(f"{example.example_id}: semantic prompt reads labels")
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"semantic selection provenance invalid: {exc}")
        else:
            for example in candidates:
                rebuilt = build_label_free_example(
                    raw_full[example.example_id],
                    tokenizer,
                    n_units=config["sample"]["n_units"],
                    max_tokens=config["segmentation"]["max_tokens"],
                    k1=config["selection"]["k1"],
                    b=config["selection"]["b"],
                )
                if rebuilt is None or unit_selection_signature(rebuilt) != unit_selection_signature(example):
                    errors.append(f"{example.example_id}: label-free selection is not reproducible")

    packet_report = packet_status(args.examples, args.packet_dir)
    if not packet_report["complete"]:
        errors.extend(f"packet: {item}" for item in packet_report["errors"])
    store = PacketStore(args.packet_dir)

    baseline_rows = list(read_jsonl(args.baseline)) if Path(args.baseline).exists() else []
    full_context_capable = 0
    if len(baseline_rows) != len(candidates):
        errors.append("full-context baseline count mismatch")
    else:
        for example, row in zip(candidates, baseline_rows):
            try:
                prediction = parse_answer(str(row["raw_prediction"]))
                f1 = token_f1(prediction, example.answer)
                valid = (
                    row["example_id"] == example.example_id
                    and row["prediction"] == prediction
                    and row["gold"] == example.answer
                    and float(row["em"]) == exact_match(prediction, example.answer)
                    and abs(float(row["f1"]) - f1) <= 1e-12
                    and int(row["context_tokens"]) == count_tokens(tokenizer, example.context)
                )
            except (KeyError, TypeError, ValueError):
                valid = False
                f1 = 0.0
            if not valid:
                errors.append(f"{example.example_id}: invalid full-context row")
            full_context_capable += int(f1 >= 0.8)

    raw_rows = _load_selected_raw(args.raw, {example.example_id for example in candidates})
    coverage_rows = list(read_jsonl(args.coverage)) if Path(args.coverage).exists() else []
    errors.extend(
        validate_fact_coverage(
            candidates, raw_rows, coverage_rows, allow_missing=True
        )
    )
    coverage_by_id = {row["example_id"]: row for row in coverage_rows}
    total_facts = sum(len(raw_supporting_facts(raw_rows[e.example_id])) for e in candidates)
    selected_facts = sum(
        fact.get("unit_id") is not None
        for row in coverage_rows
        for fact in row.get("facts", [])
    )
    top6_coverage = selected_facts / total_facts if total_facts else 0.0

    all_results: dict[str, list[ExactSearchResult]] = {}
    exact_paths: list[Path] = []
    expected_states = 3 ** config["sample"]["n_units"]
    for example in candidates:
        raw_path = Path(args.raw_exact_dir) / f"{example.example_id}.jsonl"
        fact_path = Path(args.fact_exact_dir) / f"{example.example_id}.jsonl"
        exact_paths.append(fact_path)
        try:
            raw_exact = list(read_jsonl(raw_path, ExactSearchResult))
            fact_exact = list(read_jsonl(fact_path, ExactSearchResult))
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{example.example_id}: exact read failed: {exc}")
            continue
        errors.extend(
            f"{example.example_id}: raw {item}"
            for item in validate_exact_rows(example, raw_exact)
        )
        errors.extend(
            f"{example.example_id}: fact {item}"
            for item in validate_exact_rows(example, fact_exact)
        )
        if len(raw_exact) != expected_states or len(fact_exact) != expected_states:
            errors.append(f"{example.example_id}: exact state count is not {expected_states}")
        if example.example_id not in coverage_by_id:
            errors.append(f"{example.example_id}: missing fact coverage")
            continue
        if fact_exact != rescore_results(raw_exact, coverage_by_id[example.example_id]):
            errors.append(f"{example.example_id}: fact rescore mismatch")
        if packet_report["complete"]:
            packets = store.get(example.example_id)
            for row in raw_exact:
                if row.tokens != count_tokens(tokenizer, build_representation(packets, row.state)):
                    errors.append(f"{example.example_id}/{row.state}: token mismatch")
                    break
                if abs(row.answer_f1 - token_f1(row.prediction, example.answer)) > 1e-12:
                    errors.append(f"{example.example_id}/{row.state}: answer F1 mismatch")
                    break
        all_results[example.example_id] = fact_exact

    adjacent_pairs = strict_rate_increases = 0
    reusable_any = reusable_all = transition_count = 0
    fact_declines = fact_transitions = 0
    fact_gain_pairs: list[float] = []
    all_gap_rows: list[dict[str, Any]] = []
    per_example_gaps: dict[str, list[float]] = defaultdict(list)
    expected_independent: list[dict[str, Any]] = []
    expected_nested: list[dict[str, Any]] = []
    expected_gaps: list[dict[str, Any]] = []
    for example in candidates:
        results = all_results.get(example.example_id, [])
        if not results:
            continue
        independent = independent_frontier(results, C_GRID)
        nested = best_nested_chain(results, C_GRID)
        gaps = structural_gap_rows(independent, nested)
        expected_independent.extend(independent)
        expected_nested.extend(nested)
        expected_gaps.extend(gaps)
        all_gap_rows.extend(gaps)
        for row in gaps:
            value = row["structural_gap_normalized"]
            if value is not None:
                per_example_gaps[example.example_id].append(float(value))
        feasible = [row for row in independent if row["feasible"]]
        for lower, higher in zip(feasible, feasible[1:]):
            adjacent_pairs += 1
            rate_increase = int(higher["tokens"]) > int(lower["tokens"])
            strict_rate_increases += int(rate_increase)
            fact_transitions += 1
            fact_declines += int(float(higher["fact_recall"]) < float(lower["fact_recall"]))
            if rate_increase:
                transition_count += 1
                _, low_states = optimal_states(results, float(lower["fidelity_level"]))
                _, high_states = optimal_states(results, float(higher["fidelity_level"]))
                relations = [is_nested(low, high) for low in low_states for high in high_states]
                reusable_any += int(any(relations))
                reusable_all += int(all(relations))
        row_by_level = {float(row["fidelity_level"]): row for row in feasible}
        if 0.60 in row_by_level and 0.90 in row_by_level:
            fact_gain_pairs.append(
                float(row_by_level[0.90]["fact_recall"])
                - float(row_by_level[0.60]["fact_recall"])
            )

    primary_artifacts = (
        (Path(args.output_dir) / "exact_frontier.csv", expected_independent),
        (Path(args.output_dir) / "nested_frontier.csv", expected_nested),
        (Path(args.output_dir) / "structural_gap.csv", expected_gaps),
    )
    for path, expected in primary_artifacts:
        if not path.exists() or _keyed(_csv(path)) != _keyed(expected):
            errors.append(f"stale or missing analysis artifact: {path}")

    if len(all_results) == len(candidates):
        auto_levels = attainable_levels(exact_paths)
        auto_independent: list[dict[str, Any]] = []
        auto_nested: list[dict[str, Any]] = []
        auto_gaps: list[dict[str, Any]] = []
        for example in candidates:
            results = all_results[example.example_id]
            independent = independent_frontier(results, auto_levels)
            nested = best_nested_chain(results, auto_levels)
            auto_independent.extend(independent)
            auto_nested.extend(nested)
            auto_gaps.extend(structural_gap_rows(independent, nested))
        for path, expected in (
            (Path(args.output_dir) / "exact_frontier_attainable.csv", auto_independent),
            (Path(args.output_dir) / "nested_frontier_attainable.csv", auto_nested),
            (Path(args.output_dir) / "structural_gap_attainable.csv", auto_gaps),
        ):
            if not path.exists() or _keyed(_csv(path)) != _keyed(expected):
                errors.append(f"stale or missing sensitivity artifact: {path}")

    normalized_gaps = [
        float(row["structural_gap_normalized"])
        for row in all_gap_rows
        if row["structural_gap_normalized"] is not None
    ]
    example_weighted_gap = mean(
        mean(values) for values in per_example_gaps.values()
    ) if per_example_gaps else None
    p90_gap = percentile(normalized_gaps, 0.90)
    strict_rate_fraction = strict_rate_increases / adjacent_pairs if adjacent_pairs else 0.0
    nested_reuse_any = reusable_any / transition_count if transition_count else 0.0
    nested_reuse_all = reusable_all / transition_count if transition_count else 0.0
    fact_gain = mean(fact_gain_pairs) if fact_gain_pairs else None
    fact_decline_fraction = fact_declines / fact_transitions if fact_transitions else 1.0

    checks = {
        "artifact_integrity": not errors,
        "minimum_complete_examples": len(all_results) >= thresholds["minimum_complete_examples"],
        "minimum_adjacent_feasible_pairs": adjacent_pairs >= thresholds["minimum_adjacent_feasible_pairs"],
        "minimum_strict_rate_transition_fraction": strict_rate_fraction >= thresholds["minimum_strict_rate_transition_fraction"],
        "minimum_nested_reuse_fraction": nested_reuse_any >= thresholds["minimum_nested_reuse_fraction"],
        "maximum_example_weighted_mean_structural_gap_normalized": example_weighted_gap is not None and example_weighted_gap <= thresholds["maximum_example_weighted_mean_structural_gap_normalized"],
        "maximum_structural_gap_normalized_p90": p90_gap is not None and p90_gap <= thresholds["maximum_structural_gap_normalized_p90"],
        "minimum_mean_fact_recall_gain_0_60_to_0_90": fact_gain is not None and fact_gain >= thresholds["minimum_mean_fact_recall_gain_0_60_to_0_90"],
        "maximum_fact_recall_decline_fraction": fact_decline_fraction <= thresholds["maximum_fact_recall_decline_fraction"],
        "minimum_top6_gold_fact_coverage": top6_coverage >= thresholds["minimum_top6_gold_fact_coverage"],
        "require_label_permutation_invariance": bool(manifest.get("label_permutation_invariance")),
    }
    passed = all(checks.values())
    return {
        "complete": not errors,
        "stage": f"{config['protocol']}_complete_stop_before_v1" if not errors else f"{config['protocol']}_incomplete",
        "scientific_gate_passed": passed,
        "decision": "GO_FOR_SEPARATE_HUMAN_V1_AUTHORIZATION" if passed else "NO_GO_STOP_BEFORE_V1",
        "errors": errors,
        "checks": checks,
        "metrics": {
            "examples": len(candidates),
            "full_context_f1_ge_0_8": full_context_capable,
            "exact_examples": len(all_results),
            "exact_states": sum(len(rows) for rows in all_results.values()),
            "gold_facts": total_facts,
            "selected_gold_facts": selected_facts,
            "top6_gold_fact_coverage": top6_coverage,
            "adjacent_feasible_pairs": adjacent_pairs,
            "strict_rate_increases": strict_rate_increases,
            "strict_rate_transition_fraction": strict_rate_fraction,
            "rate_increase_transitions": transition_count,
            "nested_reuse_fraction_any_tie": nested_reuse_any,
            "nested_reuse_fraction_all_ties": nested_reuse_all,
            "example_weighted_mean_structural_gap_normalized": example_weighted_gap,
            "structural_gap_normalized_p90": p90_gap,
            "positive_structural_gap_rows": sum(value > 0 for value in normalized_gaps),
            "paired_0_60_0_90_examples": len(fact_gain_pairs),
            "mean_fact_recall_gain_0_60_to_0_90": fact_gain,
            "fact_recall_decline_fraction": fact_decline_fraction,
        },
        "preregistered_config_sha256": sha256(args.config),
        "artifacts": {
            str(path): sha256(path)
            for path in (
                Path(args.config), Path(args.manifest), Path(args.examples),
                Path(args.baseline), Path(args.coverage),
                Path(args.output_dir) / "exact_frontier.csv",
                Path(args.output_dir) / "nested_frontier.csv",
                Path(args.output_dir) / "structural_gap.csv",
                Path(args.output_dir) / "exact_frontier_attainable.csv",
                Path(args.output_dir) / "nested_frontier_attainable.csv",
                Path(args.output_dir) / "structural_gap_attainable.csv",
            )
            if path.exists()
        },
        "artifact_trees": {
            args.packet_dir: tree_sha256(args.packet_dir),
            args.raw_exact_dir: tree_sha256(args.raw_exact_dir),
            args.fact_exact_dir: tree_sha256(args.fact_exact_dir),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and decide the preregistered V0.3 gate")
    parser.add_argument("--config", default="configs/v0_3_gate.json")
    parser.add_argument("--manifest", default="results/v0_3/data_manifest.json")
    parser.add_argument("--examples", default="data/units/hotpot_v0_3_candidates.jsonl")
    parser.add_argument("--raw", default="data/raw/hotpot_validation.parquet")
    parser.add_argument("--baseline", default="results/v0_3/full_context.jsonl")
    parser.add_argument("--packet-dir", default="data/packets_v0_3")
    parser.add_argument("--coverage", default="results/v0_3/fact_coverage.jsonl")
    parser.add_argument("--raw-exact-dir", default="results/v0_3/exact_search_raw")
    parser.add_argument("--fact-exact-dir", default="results/v0_3/exact_search")
    parser.add_argument("--tokenizer", default="models/Qwen3-8B")
    parser.add_argument("--output", default="results/v0_3/gate_result.json")
    parser.add_argument("--output-dir", default="results/v0_3")
    args = parser.parse_args()
    report = evaluate_gate(args)
    write_metadata(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
