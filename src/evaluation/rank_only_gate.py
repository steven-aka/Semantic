from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from src.data.schemas import read_jsonl
from src.reproducibility import sha256, write_metadata


def summarize_gate_run(summary: dict[str, Any], details: list[dict[str, Any]]) -> dict[str, float]:
    if not summary.get("complete"):
        raise ValueError("ranking evaluation is not complete")
    if len(details) != int(summary["examples"]):
        raise ValueError("ranking summary/detail example counts differ")
    highest = []
    for row in details:
        anchors = row.get("anchors", [])
        if not anchors:
            raise ValueError("ranking detail has no active anchors")
        highest.append(bool(anchors[-1]["contract_success"]))
    return {
        "active_contract_success": float(
            summary["oracle_cutoff_active_contract_success_fraction"]
        ),
        "all_active_trajectory_success": float(
            summary["oracle_cutoff_all_active_contracts_success_fraction"]
        ),
        "highest_active_anchor_success": mean(highest),
        "normalized_ranking_regret_feasible": float(
            summary["mean_oracle_cutoff_ranking_regret_normalized_feasible"]
        ),
        "identifiable_pairwise_accuracy": float(
            summary["identifiable_pairwise_accuracy"]
        ),
    }


def apply_conjunctive_gate(
    runs: Sequence[dict[str, float]], thresholds: dict[str, float]
) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one ranking run is required")
    metrics = {key: mean(run[key] for run in runs) for key in runs[0]}
    checks = {
        "active_contract_success": metrics["active_contract_success"]
        >= thresholds["minimum_mean_active_contract_success"],
        "all_active_trajectory_success": metrics["all_active_trajectory_success"]
        >= thresholds["minimum_mean_all_active_trajectory_success"],
        "highest_active_anchor_success": metrics["highest_active_anchor_success"]
        >= thresholds["minimum_mean_highest_active_anchor_success"],
        "normalized_ranking_regret_feasible": metrics[
            "normalized_ranking_regret_feasible"
        ]
        <= thresholds["maximum_mean_normalized_ranking_regret_feasible"],
        "identifiable_pairwise_accuracy": metrics["identifiable_pairwise_accuracy"]
        >= thresholds["minimum_mean_identifiable_pairwise_accuracy"],
    }
    return {
        "mean_across_seeds": metrics,
        "checks": checks,
        "scientific_gate_passed": all(checks.values()),
        "decision": "GO_IMPLEMENT_CUTOFF" if all(checks.values()) else "NO_GO_STOP_AT_RANKING",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen three-seed V2 rank-only gate")
    parser.add_argument("--config", default="configs/v2_rank_then_cut_hypothesis.json")
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("SUMMARY", "DETAILS", "TRAINING_METADATA"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    frozen_seeds = set(int(seed) for seed in config["ranking_model"]["seeds"])
    if len(args.run) != len(frozen_seeds):
        raise ValueError("rank-only gate requires exactly the three frozen seed runs")
    run_metrics = []
    artifacts = []
    observed_seeds = set()
    validation_ids: set[str] | None = None
    for summary_path, details_path, training_path in args.run:
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        details = list(read_jsonl(details_path))
        training = json.loads(Path(training_path).read_text(encoding="utf-8"))
        seed = int(training["seed"])
        if int(training["train_examples"]) != int(
            config["ranking_only_gate_before_cutoff"]["evaluated_size"]
        ):
            raise ValueError("rank-only gate may only use train2000 checkpoints")
        if summary["artifacts"]["checkpoint_metadata_sha256"] != sha256(training_path):
            raise ValueError("evaluation/checkpoint metadata hash mismatch")
        ids = {str(row["example_id"]) for row in details}
        if len(ids) != 300:
            raise ValueError("frozen rank-only gate requires validation300")
        if validation_ids is not None and ids != validation_ids:
            raise ValueError("seed runs used different ranking-validation examples")
        validation_ids = ids
        observed_seeds.add(seed)
        run_metrics.append(summarize_gate_run(summary, details))
        artifacts.append(
            {
                "seed": seed,
                "summary": summary_path,
                "summary_sha256": sha256(summary_path),
                "details": details_path,
                "details_sha256": sha256(details_path),
                "training_metadata": training_path,
                "training_metadata_sha256": sha256(training_path),
            }
        )
    if observed_seeds != frozen_seeds:
        raise ValueError("gate runs do not match the three frozen seeds")

    result = apply_conjunctive_gate(
        run_metrics, config["ranking_only_gate_before_cutoff"]
    )
    result.update(
        {
            "complete": True,
            "protocol_config": args.config,
            "protocol_config_sha256": sha256(args.config),
            "validation_examples": len(validation_ids or ()),
            "seeds": sorted(observed_seeds),
            "per_seed": run_metrics,
            "artifacts": artifacts,
        }
    )
    write_metadata(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
