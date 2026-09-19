"""Read-only H0 checkpoint replay with threshold-consistent scalar scoring."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.v17d1b_frozen_boundary_separability import stable_fold
from src.reproducibility import sha256, write_metadata
from src.training.train_v17h0_multianchor_cutoff import (
    MultiAnchorCutoff,
    PrefixData,
    PrefixDataSubset,
    candidate_oracle_index,
    evaluate_role,
    predict_all,
)


def main() -> None:
    root = Path("results/v2_rank_then_cut")
    out = root / "v17h0_multianchor_cutoff_pilot"
    original_path = out / "summary.json"
    original = json.loads(original_path.read_text())
    original_copy = out / "original_summary_before_float32_correction.json"
    if not original_copy.exists():
        write_metadata(original_copy, original)
    config_path = Path("configs/v17h0_multianchor_cutoff_pilot.json")
    config = json.loads(config_path.read_text())
    model = MultiAnchorCutoff()
    checkpoint = out / "cutoff_step400.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    rollouts = root / "v9_v8_train3163_beam8_rollouts.jsonl"
    exact = root / "candidates5000_exact"
    train = PrefixData(
        str(root / "v14_train2863_top4_candidates.jsonl"),
        str(root / "v14_train2863_v13_embeddings.pt"),
        str(rollouts),
        str(root / "v16c0_deployed_fid_audit_v8_step250/train2863_details.jsonl"),
        str(exact),
    )
    heldout_indices = [i for i, query in enumerate(train.ids) if stable_fold(query, 5) == 0]
    assert len(heldout_indices) == 606
    heldout = PrefixDataSubset(train, heldout_indices)
    heldout_prob = predict_all(model, train, heldout_indices, torch.device("cpu"))
    threshold = config["sufficiency_probability_threshold"]
    heldout_results = evaluate_role(heldout, heldout_prob, [0] * len(heldout.ids), threshold)
    heldout_results.pop("details")
    internal = PrefixData(
        str(root / "v14_internal_validation300_top4_candidates.jsonl"),
        str(root / "v14_internal_validation300_v13_embeddings.pt"),
        str(rollouts),
        str(root / "v16c0_deployed_fid_audit_v8_step250/internal_validation300_details.jsonl"),
        str(exact),
    )
    internal_prob = predict_all(model, internal, list(range(len(internal.ids))), torch.device("cpu"))
    choices = {row["example_id"]: int(row["selected_index"]) for row in read_jsonl(root / "v14_conservative_selector_seed20260918/internal_validation_selection.jsonl")}
    assert set(choices) == set(internal.ids)
    results = {
        "v8_fallback": evaluate_role(internal, internal_prob, [0] * len(internal.ids), threshold),
        "v14_frozen_selector": evaluate_role(internal, internal_prob, [choices[query] for query in internal.ids], threshold),
        "oracle_candidate_diagnostic": evaluate_role(internal, internal_prob, [candidate_oracle_index(row) for row in internal.rows], threshold),
    }
    write_jsonl(out / "internal_v14_selector_details.jsonl", results["v14_frozen_selector"].pop("details"))
    for result in results.values():
        result.pop("details", None)
    corrected = dict(original)
    corrected["heldout_v8_fallback"] = heldout_results
    corrected["internal_design_exposed"] = results
    corrected["evaluation_revision"] = {
        "reason": "Original Python scalar comparisons used float32 fidelity + 1e-12, misclassifying exact threshold values such as 0.900000; replay uses a 1e-6 tolerance consistent with exact-cache labels.",
        "checkpoint_retrained": False,
        "original_summary": original_copy.name,
        "corrected_replay_script_sha256": sha256(__file__),
        "checkpoint_sha256": sha256(checkpoint),
    }
    write_metadata(original_path, corrected)
    print(json.dumps(corrected, indent=2), flush=True)


if __name__ == "__main__":
    main()
