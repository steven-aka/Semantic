from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from src.data.schemas import ExactSearchResult, QAExample, read_jsonl
from src.evaluation.v0_3_gate import optimal_states, percentile
from src.representation.atomic_packet_store import AtomicPacketStore
from src.representation.atomic_packetizer import validate_atomic_packets
from src.representation.state_builder import build_atomic_representation
from src.representation.token_counter import count_tokens, load_tokenizer
from src.reproducibility import sha256, tree_sha256, write_metadata
from src.search.atomic_exact_search import validate_atomic_exact_rows
from src.search.atomic_nested_chain import best_binary_nested_chain
from src.search.best_nested_chain import is_nested, structural_gap_rows
from src.search.exact_frontier import independent_frontier, write_csv


def _checks(metrics: Mapping[str, Any], limits: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "artifact_integrity": metrics["errors"] == [],
        "minimum_complete_examples": metrics["examples"] >= limits["minimum_complete_examples"],
        "minimum_adjacent_feasible_pairs": metrics["adjacent_feasible_pairs"] >= limits["minimum_adjacent_feasible_pairs"],
        "minimum_strict_rate_transition_fraction": metrics["strict_rate_transition_fraction"] >= limits["minimum_strict_rate_transition_fraction"],
        "minimum_nested_reuse_fraction": metrics["nested_reuse_fraction_any_tie"] >= limits["minimum_nested_reuse_fraction"],
        "minimum_nonnested_independent_switches": metrics["nonnested_independent_switches"] >= limits["minimum_nonnested_independent_switches"],
        "maximum_mean_structural_gap_normalized": metrics["mean_structural_gap_normalized"] <= limits["maximum_mean_structural_gap_normalized"],
        "maximum_structural_gap_normalized_p90": metrics["structural_gap_normalized_p90"] <= limits["maximum_structural_gap_normalized_p90"],
        "minimum_mean_atom_recall_gain_0_60_to_0_90": metrics["mean_atom_recall_gain_0_60_to_0_90"] >= limits["minimum_mean_atom_recall_gain_0_60_to_0_90"],
        "maximum_atom_recall_decline_fraction": metrics["atom_recall_decline_fraction"] <= limits["maximum_atom_recall_decline_fraction"],
        "minimum_full_state_f1_0_8_fraction": metrics["full_state_f1_0_8_fraction"] >= limits["minimum_full_state_f1_0_8_fraction"],
        "minimum_max_f1_0_9_fraction": metrics["max_f1_0_9_fraction"] >= limits["minimum_max_f1_0_9_fraction"],
        "require_atomic_packet_partition": metrics["atomic_packet_partition_complete"] == bool(limits["require_atomic_packet_partition"]),
        "require_source_reconstruction": metrics["source_reconstruction_complete"] == bool(limits["require_source_reconstruction"]),
        "require_evidence_certificate": metrics["evidence_certificate_complete"] == bool(limits["require_evidence_certificate"]),
        "require_ceiling_precondition": metrics["ceiling_precondition_complete"] == bool(limits["require_ceiling_precondition"]),
    }


def evaluate_atomic_gate(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    examples = list(read_jsonl(args.examples, QAExample))
    annotations = {str(row["example_id"]): row for row in read_jsonl(args.annotations)}
    baseline_rows = list(read_jsonl(args.ceiling_baseline))
    baseline = {str(row["example_id"]): row for row in baseline_rows}
    tokenizer = load_tokenizer(args.tokenizer)
    store = AtomicPacketStore(args.packet_dir)
    levels = tuple(float(value) for value in config["primary_grid"])
    errors: list[str] = []

    expected = config["artifacts"]
    for name, path in (
        ("examples", args.examples),
        ("annotations", args.annotations),
        ("selection_manifest", args.selection_manifest),
        ("ceiling_baseline", args.ceiling_baseline),
    ):
        if sha256(path) != expected[f"{name}_sha256"]:
            errors.append(f"{name} changed after freeze")
    if tree_sha256(args.packet_dir) != expected["packet_tree_sha256"]:
        errors.append("packet tree changed after freeze")
    if len(examples) != int(config["sample"]["n_examples"]):
        errors.append("example count mismatch")
    for path, expected_hash in config["implementation_sha256"].items():
        if sha256(path) != expected_hash:
            errors.append(f"implementation changed after freeze: {path}")

    independent_rows: list[dict[str, object]] = []
    nested_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    gaps_by_example: dict[str, list[float]] = defaultdict(list)
    all_gaps: list[float] = []
    exact_states = packet_count = reconstruction_passes = 0
    evidence_passes = ceiling_passes = full_high = max_high = 0
    adjacent = strict = switches = nonnested = declines = 0
    rate_transitions = reusable_any = reusable_all = 0
    gains: list[float] = []

    ceiling = config["ceiling_precondition"]
    for example in examples:
        annotation = annotations.get(example.example_id)
        if annotation is None:
            errors.append(f"{example.example_id}: missing annotation")
            continue
        evidence_passes += int(bool(annotation.get("evidence_certificate_pass", False)))
        baseline_row = baseline.get(example.example_id)
        if baseline_row is None:
            errors.append(f"{example.example_id}: missing ceiling baseline")
        else:
            passes = (
                float(baseline_row["full_metrics"]["f1"]) >= float(ceiling["minimum_full_f1"])
                and float(baseline_row["empty_metrics"]["f1"]) <= float(ceiling["maximum_empty_f1"])
                and float(baseline_row["f1_context_gain"]) >= float(ceiling["minimum_context_gain"])
                and (
                    not bool(ceiling["require_nontruncated"])
                    or str(baseline_row.get("full_finish_reason")) != "length"
                )
            )
            ceiling_passes += int(passes)
        try:
            packets = store.get(example.example_id)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{example.example_id}: packet read failed: {exc}")
            continue
        packet_count += len(packets)
        if len(packets) != int(config["sample"]["packets_per_example"]):
            errors.append(
                f"{example.example_id}: packet count {len(packets)} != "
                f"{config['sample']['packets_per_example']}"
            )
        packet_errors = validate_atomic_packets(example, packets, tokenizer)
        if packet_errors:
            errors.extend(f"{example.example_id}: {error}" for error in packet_errors)
        else:
            reconstruction_passes += 1
        exact_path = Path(args.exact_dir) / f"{example.example_id}.jsonl"
        try:
            exact = list(read_jsonl(exact_path, ExactSearchResult))
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{example.example_id}: exact read failed: {exc}")
            continue
        exact_errors = validate_atomic_exact_rows(example, packets, exact)
        if exact_errors:
            errors.extend(f"{example.example_id}: {error}" for error in exact_errors)
            continue
        for row in exact:
            expected_tokens = count_tokens(
                tokenizer, build_atomic_representation(packets, row.state)
            )
            if row.tokens != expected_tokens:
                errors.append(f"{example.example_id}/{row.state}: token mismatch")
                break
        exact_states += len(exact)
        full = next(row for row in exact if row.state == (1,) * len(packets))
        full_high += int(full.fidelity >= 0.8)
        max_high += int(max(row.fidelity for row in exact) >= 0.9)

        independent = independent_frontier(exact, levels)
        nested = best_binary_nested_chain(exact, levels)
        gaps = structural_gap_rows(independent, nested)
        independent_rows.extend(independent)
        nested_rows.extend(nested)
        gap_rows.extend(gaps)
        for gap in gaps:
            value = gap["structural_gap_normalized"]
            if value is not None:
                value = float(value)
                all_gaps.append(value)
                gaps_by_example[example.example_id].append(value)
        feasible = [row for row in independent if row["feasible"]]
        for lower, higher in zip(feasible, feasible[1:]):
            adjacent += 1
            changed = lower["state"] != higher["state"]
            switches += int(changed)
            nonnested += int(changed and not is_nested(lower["state"], higher["state"]))
            strict += int(int(higher["tokens"]) > int(lower["tokens"]))
            declines += int(float(higher["fact_recall"]) < float(lower["fact_recall"]))
            if int(higher["tokens"]) > int(lower["tokens"]):
                rate_transitions += 1
                _, low_states = optimal_states(exact, float(lower["fidelity_level"]))
                _, high_states = optimal_states(exact, float(higher["fidelity_level"]))
                relations = [
                    is_nested(low, high)
                    for low in low_states
                    for high in high_states
                ]
                reusable_any += int(any(relations))
                reusable_all += int(all(relations))
        by_level = {float(row["fidelity_level"]): row for row in feasible}
        if 0.60 in by_level and 0.90 in by_level:
            gains.append(
                float(by_level[0.90]["fact_recall"])
                - float(by_level[0.60]["fact_recall"])
            )

    write_csv(Path(args.output_dir) / "independent_frontier.csv", independent_rows)
    write_csv(Path(args.output_dir) / "nested_frontier.csv", nested_rows)
    write_csv(Path(args.output_dir) / "structural_gap.csv", gap_rows)
    n = len(examples)
    metrics = {
        "errors": errors,
        "examples": len({row["example_id"] for row in independent_rows}),
        "exact_states": exact_states,
        "atomic_packets": packet_count,
        "atomic_packet_partition_complete": (
            packet_count
            == n * int(config["sample"]["packets_per_example"])
        ),
        "source_reconstruction_passes": reconstruction_passes,
        "source_reconstruction_complete": reconstruction_passes == n,
        "evidence_certificates": evidence_passes,
        "evidence_certificate_complete": evidence_passes == n,
        "ceiling_precondition_passes": ceiling_passes,
        "ceiling_precondition_complete": ceiling_passes == n,
        "adjacent_feasible_pairs": adjacent,
        "strict_rate_increases": strict,
        "strict_rate_transition_fraction": strict / adjacent if adjacent else 0.0,
        "state_switches": switches,
        "nonnested_independent_switches": nonnested,
        "nested_reuse_fraction_any_tie": (
            reusable_any / rate_transitions if rate_transitions else 0.0
        ),
        "nested_reuse_fraction_all_ties": (
            reusable_all / rate_transitions if rate_transitions else 0.0
        ),
        "paired_0_60_0_90_examples": len(gains),
        "mean_atom_recall_gain_0_60_to_0_90": mean(gains) if gains else 0.0,
        "atom_recall_decline_fraction": declines / adjacent if adjacent else 1.0,
        "full_state_f1_0_8_examples": full_high,
        "full_state_f1_0_8_fraction": full_high / n if n else 0.0,
        "max_f1_0_9_examples": max_high,
        "max_f1_0_9_fraction": max_high / n if n else 0.0,
        "mean_structural_gap_normalized": (
            mean(mean(values) for values in gaps_by_example.values())
            if gaps_by_example else 1.0
        ),
        "structural_gap_normalized_p90": percentile(all_gaps, 0.90) if all_gaps else 1.0,
        "positive_structural_gap_rows": sum(value > 0 for value in all_gaps),
    }
    checks = _checks(metrics, config["hard_gates"])
    passed = all(checks.values())
    protocol = str(config["protocol"])
    heldout = "heldout" in protocol
    return {
        "complete": not errors,
        "stage": (
            "m0_atomic_binary_heldout_complete_stop_before_v1"
            if heldout and not errors else
            "m0_atomic_binary_development_complete_stop_before_v1"
            if not errors else "m0_atomic_binary_incomplete"
        ),
        "scientific_gate_passed": passed,
        "decision": (
            "GO_V1_ATOMIC_ORDINAL_READINESS_NOT_TRAINING"
            if passed and heldout else
            "READY_TO_FREEZE_FRESH_ATOMIC_HELDOUT_NOT_V1"
            if passed else "NO_GO_REVISIT_ATOMIC_REPRESENTATION"
        ),
        "checks": checks,
        "metrics": metrics,
        "config_sha256": sha256(args.config),
        "artifacts": {
            "examples_sha256": sha256(args.examples),
            "annotations_sha256": sha256(args.annotations),
            "selection_manifest_sha256": sha256(args.selection_manifest),
            "ceiling_baseline_sha256": sha256(args.ceiling_baseline),
            "packet_tree_sha256": tree_sha256(args.packet_dir),
            "exact_tree_sha256": tree_sha256(args.exact_dir),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen binary atomic M0 gate")
    parser.add_argument("--config", required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--ceiling-baseline", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default="models/Qwen3-8B")
    args = parser.parse_args()
    result = evaluate_atomic_gate(args)
    write_metadata(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()
