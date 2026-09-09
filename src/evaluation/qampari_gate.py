from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from src.data.schemas import ExactSearchResult, QAExample, read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.evaluation.v0_3_gate import optimal_states, percentile
from src.representation.lossless_packetizer import validate_lossless_partition
from src.representation.packet_store import PacketStore
from src.representation.state_builder import build_representation
from src.representation.token_counter import count_tokens, load_tokenizer
from src.reproducibility import sha256, tree_sha256, write_metadata
from src.search.best_nested_chain import best_nested_chain, is_nested, structural_gap_rows
from src.search.exact_frontier import independent_frontier, write_csv
from src.search.qampari_exact_search import validate_qampari_exact_rows


def _checks(metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "artifact_integrity": metrics["errors"] == [],
        "minimum_complete_examples": metrics["examples"] >= thresholds["minimum_complete_examples"],
        "minimum_adjacent_feasible_pairs": metrics["adjacent_feasible_pairs"] >= thresholds["minimum_adjacent_feasible_pairs"],
        "minimum_strict_rate_transition_fraction": metrics["strict_rate_transition_fraction"] >= thresholds["minimum_strict_rate_transition_fraction"],
        "minimum_nested_reuse_fraction": metrics["nested_reuse_fraction_any_tie"] >= thresholds["minimum_nested_reuse_fraction"],
        "maximum_example_weighted_mean_structural_gap_normalized": metrics["example_weighted_mean_structural_gap_normalized"] <= thresholds["maximum_example_weighted_mean_structural_gap_normalized"],
        "maximum_structural_gap_normalized_p90": metrics["structural_gap_normalized_p90"] <= thresholds["maximum_structural_gap_normalized_p90"],
        "minimum_mean_atom_recall_gain_0_60_to_0_90": metrics["mean_atom_recall_gain_0_60_to_0_90"] >= thresholds["minimum_mean_atom_recall_gain_0_60_to_0_90"],
        "maximum_atom_recall_decline_fraction": metrics["atom_recall_decline_fraction"] <= thresholds["maximum_atom_recall_decline_fraction"],
        "minimum_full_state_f1_0_8_fraction": metrics["full_state_f1_0_8_fraction"] >= thresholds["minimum_full_state_f1_0_8_fraction"],
        "minimum_max_f1_0_9_fraction": metrics["max_f1_0_9_fraction"] >= thresholds["minimum_max_f1_0_9_fraction"],
        "minimum_nonnested_independent_switches": metrics["nonnested_independent_switches"] >= thresholds["minimum_nonnested_independent_switches"],
        "require_lossless_partition": metrics["lossless_partition_complete"] == bool(thresholds["require_lossless_partition"]),
    }


def evaluate_qampari_gate(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    examples = list(read_jsonl(args.examples, QAExample))
    annotations = {row["example_id"]: row for row in read_jsonl(args.annotations)}
    tokenizer = load_tokenizer(args.tokenizer)
    store = PacketStore(args.packet_dir)
    levels = tuple(float(value) for value in config["primary_grid"])
    errors: list[str] = []

    expected_hashes = config["sample"]
    for name, path, expected in (
        ("examples", args.examples, expected_hashes["examples_sha256"]),
        ("annotations", args.annotations, expected_hashes["annotations_sha256"]),
        ("selection manifest", args.selection_manifest, expected_hashes["selection_manifest_sha256"]),
    ):
        if sha256(path) != expected:
            errors.append(f"{name} changed after gate freeze")
    if tree_sha256(args.packet_dir) != config["packetization"]["packet_tree_sha256"]:
        errors.append("packet tree changed after gate freeze")
    if len(examples) != int(config["sample"]["n_examples"]):
        errors.append("example count mismatch")

    adjacent_pairs = strict_increases = atom_declines = 0
    rate_transitions = reusable_any = reusable_all = 0
    state_switches = nonnested_switches = 0
    atom_gains: list[float] = []
    per_example_gaps: dict[str, list[float]] = defaultdict(list)
    all_gaps: list[float] = []
    independent_rows: list[dict[str, object]] = []
    nested_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    full_state_high = max_high = exact_examples = exact_states = 0
    lossless_packets = 0

    for example in examples:
        if example.example_id not in annotations:
            errors.append(f"{example.example_id}: missing annotation")
            continue
        annotation = annotations[example.example_id]
        if len(annotation.get("answer_atoms", [])) != int(config["sample"]["answer_atoms_per_example"]):
            errors.append(f"{example.example_id}: answer atom count mismatch")
        try:
            packets = store.get(example.example_id)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{example.example_id}: packet read failed: {exc}")
            continue
        if len(packets) != len(example.units):
            errors.append(f"{example.example_id}: packet count mismatch")
            continue
        packet_valid = True
        for unit, packet in zip(example.units, packets):
            if packet.source != unit.text or packet.example_id != example.example_id:
                errors.append(f"{example.example_id}/{unit.unit_id}: packet/source mismatch")
                packet_valid = False
                continue
            partition_errors = validate_lossless_partition(packet, tokenizer)
            if partition_errors:
                errors.append(
                    f"{example.example_id}/{unit.unit_id}: " + "; ".join(partition_errors)
                )
                packet_valid = False
            else:
                lossless_packets += 1
        if not packet_valid:
            continue

        path = Path(args.exact_dir) / f"{example.example_id}.jsonl"
        try:
            results = list(read_jsonl(path, ExactSearchResult))
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{example.example_id}: exact read failed: {exc}")
            continue
        row_errors = validate_qampari_exact_rows(example, results)
        errors.extend(f"{example.example_id}: {error}" for error in row_errors)
        if row_errors:
            continue
        atoms = annotation["answer_atoms"]
        for row in results:
            measured = qampari_list_metrics(parse_list_prediction(row.prediction), atoms)
            expected_em = float(measured["correct"] == len(atoms) and measured["predicted"] == len(atoms))
            if (
                abs(row.answer_f1 - float(measured["f1"])) > 1e-12
                or abs(row.fact_recall - float(measured["recall"])) > 1e-12
                or abs(row.fidelity - float(measured["f1"])) > 1e-12
                or row.answer_em != expected_em
            ):
                errors.append(f"{example.example_id}/{row.state}: hard metric mismatch")
                break
            expected_tokens = count_tokens(tokenizer, build_representation(packets, row.state))
            if row.tokens != expected_tokens:
                errors.append(f"{example.example_id}/{row.state}: token mismatch")
                break
        exact_examples += 1
        exact_states += len(results)
        full_state = next(row for row in results if row.state == (2,) * len(example.units))
        full_state_high += int(full_state.fidelity >= 0.8)
        max_high += int(max(row.fidelity for row in results) >= 0.9)

        independent = independent_frontier(results, levels)
        nested = best_nested_chain(results, levels)
        gaps = structural_gap_rows(independent, nested)
        independent_rows.extend(independent)
        nested_rows.extend(nested)
        gap_rows.extend(gaps)
        for gap in gaps:
            value = gap["structural_gap_normalized"]
            if value is not None:
                value = float(value)
                all_gaps.append(value)
                per_example_gaps[example.example_id].append(value)
        feasible = [row for row in independent if row["feasible"]]
        for lower, higher in zip(feasible, feasible[1:]):
            adjacent_pairs += 1
            changed = lower["state"] != higher["state"]
            state_switches += int(changed)
            nonnested_switches += int(
                changed and not is_nested(lower["state"], higher["state"])
            )
            increased = int(higher["tokens"]) > int(lower["tokens"])
            strict_increases += int(increased)
            atom_declines += int(
                float(higher["fact_recall"]) < float(lower["fact_recall"])
            )
            if increased:
                rate_transitions += 1
                _, low_states = optimal_states(results, float(lower["fidelity_level"]))
                _, high_states = optimal_states(results, float(higher["fidelity_level"]))
                relations = [is_nested(low, high) for low in low_states for high in high_states]
                reusable_any += int(any(relations))
                reusable_all += int(all(relations))
        by_level = {float(row["fidelity_level"]): row for row in feasible}
        if 0.60 in by_level and 0.90 in by_level:
            atom_gains.append(
                float(by_level[0.90]["fact_recall"])
                - float(by_level[0.60]["fact_recall"])
            )

    write_csv(Path(args.output_dir) / "lossless_exact_frontier.csv", independent_rows)
    write_csv(Path(args.output_dir) / "lossless_nested_frontier.csv", nested_rows)
    write_csv(Path(args.output_dir) / "lossless_structural_gap.csv", gap_rows)
    normalized_gap_mean = (
        mean(mean(values) for values in per_example_gaps.values())
        if per_example_gaps
        else 1.0
    )
    metrics = {
        "errors": errors,
        "examples": exact_examples,
        "exact_states": exact_states,
        "lossless_packets": lossless_packets,
        "lossless_partition_complete": lossless_packets == len(examples) * 6,
        "adjacent_feasible_pairs": adjacent_pairs,
        "strict_rate_increases": strict_increases,
        "strict_rate_transition_fraction": strict_increases / adjacent_pairs if adjacent_pairs else 0.0,
        "state_switches": state_switches,
        "nonnested_independent_switches": nonnested_switches,
        "nested_reuse_fraction_any_tie": reusable_any / rate_transitions if rate_transitions else 0.0,
        "nested_reuse_fraction_all_ties": reusable_all / rate_transitions if rate_transitions else 0.0,
        "paired_0_60_0_90_examples": len(atom_gains),
        "mean_atom_recall_gain_0_60_to_0_90": mean(atom_gains) if atom_gains else 0.0,
        "atom_recall_decline_fraction": atom_declines / adjacent_pairs if adjacent_pairs else 1.0,
        "full_state_f1_0_8_examples": full_state_high,
        "full_state_f1_0_8_fraction": full_state_high / len(examples) if examples else 0.0,
        "max_f1_0_9_examples": max_high,
        "max_f1_0_9_fraction": max_high / len(examples) if examples else 0.0,
        "example_weighted_mean_structural_gap_normalized": normalized_gap_mean,
        "structural_gap_normalized_p90": percentile(all_gaps, 0.90) if all_gaps else 1.0,
        "positive_structural_gap_rows": sum(value > 0 for value in all_gaps),
    }
    checks = _checks(metrics, config["hard_gates"])
    passed = all(checks.values())
    return {
        "complete": not errors,
        "stage": "m0_qampari_lossless_development_complete_stop_before_v1" if not errors else "m0_qampari_lossless_development_incomplete",
        "scientific_gate_passed": passed,
        "decision": "READY_TO_FREEZE_FRESH_LOCKED_M0_NOT_V1" if passed else "NO_GO_STOP_BEFORE_V1",
        "checks": checks,
        "metrics": metrics,
        "config_sha256": sha256(args.config),
        "artifacts": {
            "examples_sha256": sha256(args.examples),
            "annotations_sha256": sha256(args.annotations),
            "selection_manifest_sha256": sha256(args.selection_manifest),
            "packet_tree_sha256": tree_sha256(args.packet_dir),
            "exact_tree_sha256": tree_sha256(args.exact_dir),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently verify the frozen QAMPARI M0 development gate")
    parser.add_argument("--config", default="configs/m0_qampari_lossless_dev_gate.json")
    parser.add_argument("--examples", default="data/units/qampari_m0_dev30_sentence.jsonl")
    parser.add_argument("--annotations", default="data/units/qampari_m0_dev30_sentence_annotations.jsonl")
    parser.add_argument("--selection-manifest", default="results/m0_qampari/dev_sentence_selection_manifest.json")
    parser.add_argument("--packet-dir", default="data/packets_qampari_m0_dev_sentence_lossless")
    parser.add_argument("--exact-dir", default="results/m0_qampari/dev_sentence_lossless_exact_search")
    parser.add_argument("--tokenizer", default="models/Qwen3-8B")
    parser.add_argument("--output-dir", default="results/m0_qampari")
    parser.add_argument("--output", default="results/m0_qampari/lossless_dev_gate_result.json")
    args = parser.parse_args()
    result = evaluate_qampari_gate(args)
    write_metadata(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()
