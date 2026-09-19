from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median

import torch

from src.data.schemas import write_jsonl
from src.evaluation.v17d1b_frozen_boundary_separability import stable_fold
from src.reproducibility import sha256, write_metadata
from src.training.train_v17h0_multianchor_cutoff import (
    LEVELS,
    MultiAnchorCutoff,
    PrefixData,
    PrefixDataSubset,
    cutoff_counts,
    meets_fidelity,
    predict_all,
    summarize_predictions,
)


def audit_row(data: PrefixDataSubset, probabilities: torch.Tensor, index: int, threshold: float) -> dict:
    active = data.active[index]
    cutoffs = cutoff_counts(probabilities[index, 0], active, threshold)
    prefix_fidelity = data.fidelity[index, 0].tolist()
    result = {"example_id": data.ids[index], "cutoffs": cutoffs, "anchors": []}
    for anchor, level in enumerate(LEVELS):
        if not bool(active[anchor]):
            result["anchors"].append(None)
            continue
        chosen = cutoffs[anchor]
        assert chosen is not None
        previous = 0 if anchor == 0 else next((cutoffs[j] for j in range(anchor - 1, -1, -1) if cutoffs[j] is not None), 0)
        assert previous is not None
        all_success = [t for t, fidelity in enumerate(prefix_fidelity) if meets_fidelity(fidelity, level)]
        available = [t for t in all_success if t >= previous]
        earliest = available[0] if available else None
        achieved = meets_fidelity(prefix_fidelity[chosen], level)
        if achieved:
            category = "success_exact_first" if chosen == earliest else "success_later"
        elif not all_success:
            category = "same_order_infeasible"
        elif not available:
            category = "blocked_by_prior_cutoff"
        elif chosen < earliest:
            category = "early_failure"
        else:
            category = "late_nonmonotone_failure"
        anchor_probability = probabilities[index, 0, :, anchor]
        result["anchors"].append({
            "level": level,
            "category": category,
            "chosen": chosen,
            "previous": previous,
            "earliest_success_after_previous": earliest,
            "earliest_success_anywhere": all_success[0] if all_success else None,
            "chosen_fidelity": prefix_fidelity[chosen],
            "chosen_probability": float(anchor_probability[chosen]),
            "earliest_success_probability": float(anchor_probability[earliest]) if earliest is not None else None,
            "max_probability_before_earliest": float(anchor_probability[previous:earliest].max()) if earliest is not None and earliest > previous else None,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v17cut_a0_fixed_trajectory_stop_audit.json")
    parser.add_argument("--checkpoint", default="results/v2_rank_then_cut/v17h0_multianchor_cutoff_pilot/cutoff_step400.pt")
    parser.add_argument("--candidates", default="results/v2_rank_then_cut/v14_train2863_top4_candidates.jsonl")
    parser.add_argument("--embeddings", default="results/v2_rank_then_cut/v14_train2863_v13_embeddings.pt")
    parser.add_argument("--rollouts", default="results/v2_rank_then_cut/v9_v8_train3163_beam8_rollouts.jsonl")
    parser.add_argument("--oracle", default="results/v2_rank_then_cut/v16c0_deployed_fid_audit_v8_step250/train2863_details.jsonl")
    parser.add_argument("--exact-dir", default="results/v2_rank_then_cut/candidates5000_exact")
    parser.add_argument("--output-dir", default="results/v2_rank_then_cut/v17cut_a0_fixed_trajectory_stop_audit")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_READ_ONLY":
        raise ValueError("protocol not frozen")
    data = PrefixData(args.candidates, args.embeddings, args.rollouts, args.oracle, args.exact_dir)
    indices = [i for i, query in enumerate(data.ids) if stable_fold(query, 5) == 0]
    if len(indices) != 606:
        raise ValueError("unexpected H0 holdout population")
    subset = PrefixDataSubset(data, indices)
    model = MultiAnchorCutoff()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    probabilities = predict_all(model, data, indices, torch.device("cpu"))
    threshold = config["primary_threshold"]
    rows = [audit_row(subset, probabilities, i, threshold) for i in range(len(indices))]
    counts = {}
    for anchor, level in enumerate(LEVELS):
        present = [row["anchors"][anchor] for row in rows if row["anchors"][anchor] is not None]
        categories = Counter(item["category"] for item in present)
        failed_recoverable = [item for item in present if item["category"] in {"blocked_by_prior_cutoff", "early_failure", "late_nonmonotone_failure"}]
        counts[str(level)] = {
            "active": len(present),
            "categories": dict(categories),
            "recoverable_same_order_failures": len(failed_recoverable),
            "median_probability_at_first_available_success_for_failures": median(item["earliest_success_probability"] for item in failed_recoverable if item["earliest_success_probability"] is not None) if any(item["earliest_success_probability"] is not None for item in failed_recoverable) else None,
        }
    scan = []
    for value in config["sweep_thresholds"]:
        result = summarize_predictions(subset, probabilities, [0] * len(indices), value)
        scan.append({"threshold": value, "success_090": result["per_anchor"][3]["successes"], "complete": result["complete"], "regret_if_complete": result["mean_regret_if_complete"]})
    main_result = summarize_predictions(subset, probabilities, [0] * len(indices), threshold)
    if main_result["per_anchor"][3]["successes"] != 510 or main_result["complete"] != 360:
        raise ValueError("H0 holdout reproduction failed")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_query.jsonl", rows)
    summary = {
        "decision": "DIAGNOSTIC_ONLY_NO_THRESHOLD_SELECTED",
        "population": len(indices),
        "baseline_090": 510,
        "oracle_same_order_090": 567,
        "baseline_complete": 360,
        "oracle_same_order_complete": 552,
        "error_categories": counts,
        "threshold_scan_same_holdout_optimistic": scan,
        "best_scanned_090_descriptive_only": max(item["success_090"] for item in scan),
        "sealed_roles_used": [],
        "artifacts": {"config_sha256": sha256(args.config), "checkpoint_sha256": sha256(args.checkpoint), "candidates_sha256": sha256(args.candidates)},
    }
    write_metadata(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
