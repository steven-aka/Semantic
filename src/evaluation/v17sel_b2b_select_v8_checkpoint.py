from __future__ import annotations

import json
import math
from pathlib import Path

from src.reproducibility import sha256, write_metadata


def main() -> None:
    root = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
    config_path = Path("configs/v17sel_b2b_lineage_clean_pool_gate.json")
    config = json.loads(config_path.read_text())
    rule = config["v8_inner_validation_checkpoint_selection"]
    selection_path = root / "v8_run/selected_checkpoint.json"
    original = json.loads(selection_path.read_text())
    records = original["checkpoints"]
    if [record["step"] for record in records] != [250, 500, 750]:
        raise ValueError("V8 checkpoint set differs from frozen protocol")
    scored = []
    for record in records:
        summary = record["summary"]
        anchor = summary["per_level"]["0.9"]
        n = int(anchor["examples"])
        if n != 550:
            raise ValueError(f"inner-validation population changed: {n}")
        success90 = int(anchor["contract_successes"])
        complete = float(summary["oracle_cutoff_all_active_contracts_success_fraction"])
        regret = float(summary["mean_oracle_cutoff_ranking_regret_normalized_feasible"])
        threshold = math.ceil(float(rule["fidelity_090_min_fraction"]) * n)
        gates = (
            success90 >= threshold,
            complete >= float(rule["complete_min_fraction"]),
            regret <= float(rule["max_normalized_regret"]),
        )
        key = (int(all(gates)), sum(gates), success90, complete, -regret, -float(summary["validation_action_loss"]))
        scored.append({"step": record["step"], "checkpoint": record["checkpoint"], "gate_success_threshold": threshold, "gates": gates, "selection_key": key})
    selected = max(scored, key=lambda item: item["selection_key"])
    output = {
        "decision": "V8_LINEAGE_CLEAN_SELECTED_FOR_DOWNSTREAM",
        "selected": selected,
        "all_checkpoints": scored,
        "training_script_selected_checkpoint_ignored": True,
        "holdout_read": False,
        "development_read_by_sel_b2b": False,
        "artifacts": {"config_sha256": sha256(config_path), "v8_selection_sha256": sha256(selection_path)},
    }
    write_metadata(root / "v8_clean_selected_checkpoint.json", output)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
