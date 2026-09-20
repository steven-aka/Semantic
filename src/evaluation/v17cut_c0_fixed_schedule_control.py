"""Quality-first fixed-depth schedule control on clean V8 orders."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import mean

import numpy as np

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.build_v17sel_c0_train_candidates import exact_values
from src.data.schemas import read_jsonl
from src.reproducibility import sha256, write_metadata
from src.training.train_v17cut_b0_structured_fixed_v8 import prefix_masks
from src.training.train_v17h0_multianchor_cutoff import LEVELS, FIDELITY_EPS


def fixed_metrics(depths: tuple[int, ...], fidelity: np.ndarray, tokens: np.ndarray,
                  active: np.ndarray, full_tokens: np.ndarray) -> dict:
    chosen_fidelity = fidelity[:, depths]
    hit = chosen_fidelity + FIDELITY_EPS >= np.asarray(LEVELS)[None, :]
    hit &= active
    complete = np.all(hit | ~active, axis=1)
    spent = (tokens[:, depths] * active).sum(axis=1)
    cost_fraction = spent / (active.sum(axis=1) * full_tokens)
    return {
        "queries": len(fidelity),
        "anchor_eligible": active.sum(axis=0).astype(int).tolist(),
        "anchor_success": hit.sum(axis=0).astype(int).tolist(),
        "complete": int(complete.sum()),
        "mean_normalized_context_cost_all_queries": float(cost_fraction.mean()),
        "mean_normalized_context_cost_complete_only": float(cost_fraction[complete].mean()) if complete.any() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--cut-b0-decisions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_TRAIN_ONLY_SELECTION_THEN_ONE_EXPOSED_EVALUATION":
        raise ValueError("unfrozen fixed schedule protocol")
    rows = list(read_jsonl(args.candidates))
    if len(rows) != 2032:
        raise ValueError("unexpected candidate population")
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    fidelity_rows, token_rows, active_rows = [], [], []
    for position, row in enumerate(rows, 1):
        masks = prefix_masks(orders[row["example_id"]])
        fidelity, tokens = exact_values(Path(args.exact_dir) / f"{row['example_id']}.jsonl")
        fidelity_rows.append([fidelity[mask] for mask in masks])
        token_rows.append([tokens[mask] for mask in masks])
        active_rows.append([level in row["attainable_levels"] for level in LEVELS])
        if position % 500 == 0:
            print(json.dumps({"loaded": position, "total": len(rows)}), flush=True)
    fidelity = np.asarray(fidelity_rows, dtype=np.float32)
    tokens = np.asarray(token_rows, dtype=np.float64)
    active = np.asarray(active_rows, dtype=bool)
    full_tokens = np.asarray([row["full_tokens"] for row in rows], dtype=np.float64)
    train = np.asarray([i for i, row in enumerate(rows) if fold(row["example_id"]) != 4])
    exposed = np.asarray([i for i, row in enumerate(rows) if fold(row["example_id"]) == 4])
    if len(train) != 1421 or len(exposed) != 611:
        raise ValueError("unexpected split")
    choices = itertools.combinations_with_replacement(range(13), 5)
    best_key = None
    best_depths = None
    best_train = None
    for depths in choices:
        metrics = fixed_metrics(depths, fidelity[train], tokens[train], active[train], full_tokens[train])
        key = (metrics["complete"], metrics["anchor_success"][3],
               -metrics["mean_normalized_context_cost_all_queries"], tuple(-depth for depth in depths))
        if best_key is None or key > best_key:
            best_key, best_depths, best_train = key, depths, metrics
    assert best_depths is not None
    # The design-exposed fold is evaluated only after the single schedule is frozen.
    exposed_metrics = fixed_metrics(best_depths, fidelity[exposed], tokens[exposed], active[exposed], full_tokens[exposed])
    b0 = {row["example_id"]: row for row in read_jsonl(args.cut_b0_decisions)}
    if set(b0) != {rows[i]["example_id"] for i in exposed}:
        raise ValueError("CUT-B0 decisions do not match exposed role")
    repair = breakage = 0
    complete_gain = complete_break = 0
    old_cost_fraction = []
    paired_complete_cost_delta = []
    selected_regret_complete = []
    for index in exposed:
        row = rows[index]
        query = row["example_id"]
        old = b0[query]
        active_count = int(active[index].sum())
        denominator = active_count * full_tokens[index]
        old_cost_fraction.append(old["cumulative_tokens"] / denominator)
        new_cost = sum(tokens[index, best_depths[anchor]] for anchor in range(5) if active[index, anchor])
        new_hit = fidelity[index, best_depths[3]] + FIDELITY_EPS >= .9
        new_complete = all(not active[index, anchor] or fidelity[index, best_depths[anchor]] + FIDELITY_EPS >= level
                           for anchor, level in enumerate(LEVELS))
        old_depth = old["stop_vector"][3]
        old_090 = fidelity[index, old_depth] + FIDELITY_EPS >= .9
        repair += int(new_hit and not old_090)
        breakage += int(old_090 and not new_hit)
        complete_gain += int(new_complete and not old["complete"])
        complete_break += int(old["complete"] and not new_complete)
        if new_complete:
            oracle_cost = row["candidates"][0]["oracle_ordered_complete_cumulative_tokens"]
            selected_regret_complete.append((new_cost - oracle_cost) / denominator)
        if new_complete and old["complete"]:
            paired_complete_cost_delta.append((new_cost - old["cumulative_tokens"]) / denominator)
    result = {
        "protocol": config["protocol"], "selected_depths": list(best_depths),
        "selection_role": "train folds0-3", "train_metrics": best_train,
        "evaluation_role": "design-exposed fold4, not independent confirmation",
        "exposed_metrics": exposed_metrics,
        "exposed_mean_normalized_regret_on_complete": mean(selected_regret_complete),
        "cut_b0_mean_normalized_context_cost_all_queries": mean(old_cost_fraction),
        "paired_vs_cut_b0": {"repair_090": repair, "break_090": breakage,
                             "complete_gain": complete_gain, "complete_break": complete_break,
                             "both_complete_queries": len(paired_complete_cost_delta),
                             "mean_paired_normalized_context_cost_delta_both_complete": mean(paired_complete_cost_delta)},
        "holdout_581_read": False, "internal300_read": False,
        "development_read": False, "confirmation_read": False, "new_target_calls": 0,
        "artifacts": {"config_sha256": sha256(args.config), "candidates_sha256": sha256(args.candidates),
                      "rollouts_sha256": sha256(args.rollouts), "cut_b0_decisions_sha256": sha256(args.cut_b0_decisions)},
    }
    write_metadata(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
