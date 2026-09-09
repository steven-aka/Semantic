from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data.schemas import ExactSearchResult, QAExample, read_jsonl
from src.evaluation.artifact_status import packet_status, validate_exact_rows
from src.evaluation.data_audit import sha256
from src.evaluation.fact_coverage import (
    _load_selected_raw,
    map_facts_to_units,
    raw_supporting_facts,
    validate_fact_coverage,
)
from src.evaluation.frontier_diagnostics import diagnose_frontier
from src.evaluation.qa_metrics import exact_match, token_f1
from src.evaluation.rescore_fact_fidelity import rescore_results
from src.representation.packet_store import PacketStore
from src.representation.state_builder import build_representation
from src.representation.token_counter import count_tokens, load_tokenizer
from src.reproducibility import write_metadata
from src.search.best_nested_chain import best_nested_chain, structural_gap_rows
from src.search.exact_frontier import C_GRID, attainable_levels, independent_frontier
from src.target.answer_parser import parse_answer


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _state(value: Any) -> tuple[int, ...] | None:
    if value in (None, "", "None"):
        return None
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return tuple(int(item) for item in value)


def _number(value: Any) -> float | None:
    return None if value in (None, "", "None") else float(value)


def _frontier_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["example_id"],
        float(row["fidelity_level"]),
        str(row["feasible"]).lower() == "true",
        _number(row["tokens"]),
        _state(row["state"]),
        _number(row["achieved_fidelity"]),
        _number(row.get("answer_f1")),
        _number(row.get("fact_recall")),
    )


def _gap_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["example_id"],
        float(row["fidelity_level"]),
        _number(row["independent_tokens"]),
        _number(row["nested_tokens"]),
        _number(row["structural_gap"]),
        _number(row["structural_gap_normalized"]),
    )


def _keyed_signatures(
    rows: Sequence[Mapping[str, Any]], signature: Any
) -> dict[tuple[str, float], tuple[Any, ...]]:
    values = [signature(row) for row in rows]
    return {(str(value[0]), float(value[1])): value for value in values}


def _metadata_field(path: Path, field: str) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle).get(field)


def verify(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    candidates = list(read_jsonl(args.candidates, QAExample))
    eligible = list(read_jsonl(args.eligible, QAExample))
    candidate_ids = [example.example_id for example in candidates]
    eligible_ids = [example.example_id for example in eligible]
    require(len(candidate_ids) == len(set(candidate_ids)), "duplicate candidate ids")
    require(len(eligible_ids) == len(set(eligible_ids)), "duplicate eligible ids")
    require(set(eligible_ids) <= set(candidate_ids), "eligible ids are not a candidate subset")

    raw_rows = _load_selected_raw(args.raw, set(candidate_ids))
    require(len(raw_rows) == len(candidates), "raw rows missing for candidates")
    for example in candidates:
        require(len(example.units) == 6, f"{example.example_id}: expected six units")
        facts = map_facts_to_units(example, raw_supporting_facts(raw_rows[example.example_id]))
        fact_units = [int(fact["unit_id"]) for fact in facts]
        require(3 <= len(facts) <= 5, f"{example.example_id}: expected 3--5 facts")
        require(len(fact_units) == len(set(fact_units)), f"{example.example_id}: facts share a unit")
        require(
            {unit.unit_id for unit in example.units if unit.supporting} == set(fact_units),
            f"{example.example_id}: supporting flags do not match exact gold facts",
        )

    baseline = list(read_jsonl(args.baseline))
    require(
        [row.get("example_id") for row in baseline] == candidate_ids,
        "baseline ids/order do not match candidates",
    )
    tokenizer = load_tokenizer(args.tokenizer)
    expected_eligible: list[str] = []
    for example, row in zip(candidates, baseline):
        prediction = parse_answer(str(row.get("raw_prediction", "")))
        f1 = token_f1(prediction, example.answer)
        em = exact_match(prediction, example.answer)
        require(prediction == row.get("prediction"), f"{example.example_id}: parsed answer mismatch")
        require(row.get("gold") == example.answer, f"{example.example_id}: gold answer mismatch")
        require(abs(float(row.get("f1", -1)) - f1) <= 1e-12, f"{example.example_id}: baseline F1 mismatch")
        require(abs(float(row.get("em", -1)) - em) <= 1e-12, f"{example.example_id}: baseline EM mismatch")
        require(
            int(row.get("context_tokens", -1)) == count_tokens(tokenizer, example.context),
            f"{example.example_id}: baseline context-token mismatch",
        )
        expected_flag = f1 >= 0.8
        require(row.get("eligible_for_supervision") is expected_flag, f"{example.example_id}: eligibility mismatch")
        if expected_flag:
            expected_eligible.append(example.example_id)
    require(eligible_ids == expected_eligible, "eligible file does not match baseline filter")

    audit_path = Path(args.data_audit)
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    require(audit.get("input_sha256") == sha256(args.candidates), "candidate data-audit hash is stale")
    require(audit.get("empty_units") == 0, "data audit reports empty units")
    require(audit.get("units_over_max_tokens") == 0, "data audit reports over-limit units")

    packet_report = packet_status(args.eligible, args.packet_dir)
    require(bool(packet_report["complete"]), "packet validation failed: " + "; ".join(packet_report["errors"]))
    store = PacketStore(args.packet_dir)
    for example in eligible:
        for packet in store.get(example.example_id):
            require(packet.source_tokens == count_tokens(tokenizer, packet.source), f"{example.example_id}/{packet.unit_id}: source-token mismatch")
            require(packet.gist_tokens == count_tokens(tokenizer, packet.gist), f"{example.example_id}/{packet.unit_id}: gist-token mismatch")
            require(packet.residual_tokens == count_tokens(tokenizer, packet.residual), f"{example.example_id}/{packet.unit_id}: residual-token mismatch")

    eligible_raw = {example.example_id: raw_rows[example.example_id] for example in eligible}
    coverage = list(read_jsonl(args.coverage))
    errors.extend(validate_fact_coverage(eligible, eligible_raw, coverage))
    coverage_by_id = {row["example_id"]: row for row in coverage}

    exact_paths = [Path(args.exact_dir) / f"{example.example_id}.jsonl" for example in eligible]
    all_results: dict[str, list[ExactSearchResult]] = {}
    for example, raw_path in zip(eligible, exact_paths):
        fact_path = Path(args.fact_exact_dir) / raw_path.name
        try:
            raw_results = list(read_jsonl(raw_path, ExactSearchResult))
            fact_results = list(read_jsonl(fact_path, ExactSearchResult))
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{example.example_id}: exact-search read failed: {exc}")
            continue
        errors.extend(f"{example.example_id}: raw {error}" for error in validate_exact_rows(example, raw_results))
        errors.extend(f"{example.example_id}: fact-aware {error}" for error in validate_exact_rows(example, fact_results))
        expected_fact = rescore_results(raw_results, coverage_by_id[example.example_id])
        require(fact_results == expected_fact, f"{example.example_id}: fact-aware rescore is stale")
        packets = store.get(example.example_id)
        for row in raw_results:
            require(row.answer_f1 == token_f1(row.prediction, example.answer), f"{example.example_id}/{row.state}: answer F1 mismatch")
            require(row.answer_em == exact_match(row.prediction, example.answer), f"{example.example_id}/{row.state}: answer EM mismatch")
            expected_tokens = count_tokens(tokenizer, build_representation(packets, row.state))
            require(row.tokens == expected_tokens, f"{example.example_id}/{row.state}: representation-token mismatch")
        all_results[example.example_id] = fact_results

    def verify_frontiers(suffix: str, levels: Sequence[float]) -> dict[str, Any]:
        expected_independent: list[dict[str, Any]] = []
        expected_nested: list[dict[str, Any]] = []
        expected_gap: list[dict[str, Any]] = []
        for example in eligible:
            results = all_results.get(example.example_id, [])
            independent = independent_frontier(results, levels)
            nested = best_nested_chain(results, levels)
            expected_independent.extend(independent)
            expected_nested.extend(nested)
            expected_gap.extend(structural_gap_rows(independent, nested))
        actual_independent = _csv(Path(args.output_dir) / f"exact_frontier{suffix}.csv")
        actual_nested = _csv(Path(args.output_dir) / f"nested_frontier{suffix}.csv")
        actual_gap = _csv(Path(args.output_dir) / f"structural_gap{suffix}.csv")
        require(
            _keyed_signatures(actual_independent, _frontier_signature)
            == _keyed_signatures(expected_independent, _frontier_signature),
            f"{suffix or 'primary'} independent frontier is stale",
        )
        require(
            _keyed_signatures(actual_nested, _frontier_signature)
            == _keyed_signatures(expected_nested, _frontier_signature),
            f"{suffix or 'primary'} nested frontier is stale",
        )
        require(
            _keyed_signatures(actual_gap, _gap_signature)
            == _keyed_signatures(expected_gap, _gap_signature),
            f"{suffix or 'primary'} structural gaps are stale",
        )
        diagnostic = diagnose_frontier(actual_independent, actual_gap)
        diagnostic_path = Path(args.output_dir) / f"diagnostics{suffix}.json"
        with diagnostic_path.open("r", encoding="utf-8") as handle:
            stored_diagnostic = json.load(handle)
        require(diagnostic == stored_diagnostic, f"{suffix or 'primary'} diagnostics are stale")
        return diagnostic

    primary = verify_frontiers("", C_GRID)
    auto = verify_frontiers("_attainable", attainable_levels(exact_paths))
    require(primary.get("assessment") == "structural_gap_identifiable", "primary grid is not identifiable")
    require(primary.get("positive_structural_gap_rows", 0) > 0, "primary grid has no positive gap")
    require(auto.get("positive_structural_gap_rows", 0) > 0, "attainable grid has no positive gap")

    metadata_checks = [
        (Path(args.baseline).with_suffix(".metadata.json"), "input_path", args.candidates),
        (Path(args.packet_dir) / "metadata.json", "input_path", args.eligible),
        (Path(args.output_dir) / "fact_coverage.metadata.json", "source_examples", args.eligible),
        (Path(args.exact_dir) / "metadata.json", "source_examples", args.eligible),
        (Path(args.fact_exact_dir) / "metadata.json", "source_examples", args.eligible),
    ]
    for path, field, expected in metadata_checks:
        require(path.exists(), f"missing metadata: {path}")
        if path.exists():
            require(_metadata_field(path, field) == str(expected), f"{path}: {field} mismatch")

    key_paths = [
        Path(args.candidates), Path(args.eligible), Path(args.baseline), Path(args.coverage),
        Path(args.output_dir) / "exact_frontier.csv",
        Path(args.output_dir) / "nested_frontier.csv",
        Path(args.output_dir) / "structural_gap.csv",
        Path(args.output_dir) / "diagnostics.json",
        Path(args.output_dir) / "exact_frontier_attainable.csv",
        Path(args.output_dir) / "nested_frontier_attainable.csv",
        Path(args.output_dir) / "structural_gap_attainable.csv",
        Path(args.output_dir) / "diagnostics_attainable.json",
    ]
    return {
        "complete": not errors,
        "stage": "v0_2_complete_stop_before_v1",
        "errors": errors,
        "counts": {
            "candidates": len(candidates),
            "eligible": len(eligible),
            "packets": sum(len(store.get(example.example_id)) for example in eligible),
            "gold_facts": sum(len(row["facts"]) for row in coverage),
            "exact_states": sum(len(rows) for rows in all_results.values()),
        },
        "primary_diagnostics": primary,
        "attainable_diagnostics": auto,
        "sha256": {str(path): sha256(path) for path in key_paths if path.exists()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end V0.2 artifact verifier")
    parser.add_argument("--candidates", default="data/units/hotpot_fact_atomic_candidates.jsonl")
    parser.add_argument("--eligible", default="data/units/hotpot_fact_atomic_eligible.jsonl")
    parser.add_argument("--raw", default="data/raw/hotpot_validation.parquet")
    parser.add_argument("--baseline", default="results/v0_2/full_context.jsonl")
    parser.add_argument("--data-audit", default="results/v0_2_data_audit.json")
    parser.add_argument("--packet-dir", default="data/packets_v0_2")
    parser.add_argument("--coverage", default="results/v0_2/fact_coverage.jsonl")
    parser.add_argument("--exact-dir", default="results/v0_2/exact_search_raw")
    parser.add_argument("--fact-exact-dir", default="results/v0_2/exact_search")
    parser.add_argument("--output-dir", default="results/v0_2")
    parser.add_argument("--tokenizer", default="models/Qwen3-8B")
    parser.add_argument("--output", default="results/v0_2/verification.json")
    args = parser.parse_args()
    report = verify(args)
    write_metadata(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
