from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.schemas import read_jsonl
from src.evaluation.rank_failure_audit import (
    clopper_pearson_lower,
    minimum_successes_for_lower_bound,
)
from src.evaluation.rank_only_gate import apply_conjunctive_gate, summarize_gate_run
from src.reproducibility import sha256, write_metadata


def decide_listwise_revision(
    run_records: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    if not run_records:
        raise ValueError("listwise decision requires at least one run")
    original_thresholds = json.loads(
        Path(config["parent_protocol"]).read_text(encoding="utf-8")
    )["ranking_only_gate_before_cutoff"]
    original_gate = apply_conjunctive_gate(
        [record["gate_metrics"] for record in run_records], original_thresholds
    )
    familywise_alpha = 0.05
    fidelity_families = 5
    per_anchor_alpha = familywise_alpha / fidelity_families
    required_lower_bound = 0.90
    risk_rows = []
    eligible = []
    for record in run_records:
        anchors = [
            anchor
            for row in record["details"]
            for anchor in row["anchors"]
            if float(anchor["fidelity_level"]) == 0.90
        ]
        successes = sum(bool(anchor["contract_success"]) for anchor in anchors)
        examples = len(anchors)
        required = minimum_successes_for_lower_bound(
            examples, required_lower_bound, per_anchor_alpha
        )
        lower = clopper_pearson_lower(successes, examples, per_anchor_alpha)
        passed = successes >= required
        risk_rows.append(
            {
                "seed": record["seed"],
                "successes": successes,
                "examples": examples,
                "required_successes": required,
                "bonferroni_clopper_pearson_lower": lower,
                "passed": passed,
            }
        )
        if passed:
            eligible.append(record)
    selected = min(
        eligible,
        key=lambda record: (record["best_validation_listwise_nll"], record["seed"]),
        default=None,
    )
    passed = bool(original_gate["scientific_gate_passed"] and selected is not None)
    return {
        "complete": True,
        "status": "post_hoc_development_decision_not_a_new_preregistered_gate",
        "original_rank_gate_recheck": original_gate,
        "risk_headroom": {
            "familywise_alpha": familywise_alpha,
            "fidelity_families": fidelity_families,
            "per_anchor_alpha": per_anchor_alpha,
            "required_contract_lower_bound": required_lower_bound,
            "runs": risk_rows,
            "at_least_one_seed_passed": bool(eligible),
        },
        "selected_seed": selected["seed"] if selected else None,
        "selected_checkpoint": selected["checkpoint"] if selected else None,
        "selected_validation_listwise_nll": (
            selected["best_validation_listwise_nll"] if selected else None
        ),
        "decision": "GO_FREEZE_CUTOFF" if passed else "STOP_V2_1_RANKING",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen V2.1 development decision")
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("SUMMARY", "DETAILS", "TRAINING_METADATA"),
        required=True,
    )
    parser.add_argument(
        "--config", default="configs/v2_1_listwise_ranking_amendment.json"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    expected_seeds = set(config["preserved"]["seeds"])
    records = []
    artifacts = []
    observed_seeds = set()
    expected_ids: set[str] | None = None
    for summary_path, details_path, training_path in args.run:
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        details = list(read_jsonl(details_path))
        training = json.loads(Path(training_path).read_text(encoding="utf-8"))
        seed = int(training["seed"])
        if training["stage"] != "v2_1_exact_partial_order_listwise_training":
            raise ValueError("decision received a non-listwise training artifact")
        if summary["artifacts"]["checkpoint_metadata_sha256"] != sha256(training_path):
            raise ValueError("evaluation and checkpoint metadata hashes differ")
        ids = {row["example_id"] for row in details}
        if expected_ids is not None and ids != expected_ids:
            raise ValueError("listwise runs evaluated different example IDs")
        expected_ids = ids
        observed_seeds.add(seed)
        checkpoint = str(Path(training_path).parent)
        records.append(
            {
                "seed": seed,
                "checkpoint": checkpoint,
                "best_validation_listwise_nll": float(training["best_validation_loss"]),
                "gate_metrics": summarize_gate_run(summary, details),
                "details": details,
            }
        )
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
    if observed_seeds != expected_seeds or len(records) != len(expected_seeds):
        raise ValueError("decision requires exactly the three frozen V2.1 seeds")
    if len(expected_ids or ()) != 300:
        raise ValueError("decision requires the consumed ranking-validation300")

    output = decide_listwise_revision(records, config)
    output.update(
        {
            "config": args.config,
            "config_sha256": sha256(args.config),
            "seeds": sorted(observed_seeds),
            "validation_examples": len(expected_ids or ()),
            "artifacts": artifacts,
        }
    )
    write_metadata(args.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
