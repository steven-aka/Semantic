from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.data.schemas import read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.training.train_v17h0_multianchor_cutoff import meets_fidelity


def full_lattice_090(path: Path) -> bool:
    seen = set()
    feasible = False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            mask = state_to_mask(row["state"])
            if mask in seen:
                raise ValueError(f"duplicate lattice mask in {path}")
            seen.add(mask)
            feasible |= meets_fidelity(float(row["fidelity"]), 0.90)
    if len(seen) != 4096:
        raise ValueError(f"incomplete lattice: {path}: {len(seen)}")
    return feasible


def main() -> None:
    root = Path("results/v2_rank_then_cut")
    config = Path("configs/v17sel_a0_candidate_ceiling_audit.json")
    if json.loads(config.read_text())["status"] != "FROZEN_READ_ONLY":
        raise ValueError("unfrozen protocol")
    train_path = root / "v14_train2863_top4_candidates.jsonl"
    internal_path = root / "v14_internal_validation300_top4_candidates.jsonl"
    selection_path = root / "v14_conservative_selector_seed20260918/internal_validation_selection.jsonl"
    exact_dir = root / "candidates5000_exact"
    train = list(read_jsonl(train_path))
    internal = list(read_jsonl(internal_path))
    if len(train) != 2863 or len(internal) != 300:
        raise ValueError("unexpected population")
    choices = {row["example_id"]: int(row["selected_index"]) for row in read_jsonl(selection_path)}
    if set(choices) != {row["example_id"] for row in internal}:
        raise ValueError("selector population mismatch")
    train_counts = Counter()
    for row in train:
        candidates = row["candidates"]
        train_counts["v8_fallback_oracle_090"] += int(candidates[0]["success"][3])
        train_counts["five_candidate_oracle_090"] += int(any(candidate["success"][3] for candidate in candidates))
    rows = []
    counts = Counter()
    for row in internal:
        query = row["example_id"]
        candidates = row["candidates"]
        choice = choices[query]
        if not 0 <= choice < len(candidates):
            raise ValueError("invalid selected candidate index")
        selected_good = bool(candidates[choice]["success"][3])
        pool_good = any(candidate["success"][3] for candidate in candidates)
        fallback_good = bool(candidates[0]["success"][3])
        if selected_good:
            category = "chosen_order_reachable"
        elif pool_good:
            category = "selector_miss_existing_candidate"
        else:
            lattice_good = full_lattice_090(exact_dir / f"{query}.jsonl")
            category = "candidate_pool_miss_global_reachable" if lattice_good else "full_lattice_infeasible"
        counts[category] += 1
        counts["v8_fallback_oracle_090"] += int(fallback_good)
        counts["five_candidate_oracle_090"] += int(pool_good)
        rows.append({
            "example_id": query,
            "selected_index": choice,
            "selected_order_reachable_090": selected_good,
            "five_candidate_pool_reachable_090": pool_good,
            "v8_fallback_reachable_090": fallback_good,
            "category": category,
        })
    if counts["chosen_order_reachable"] != 280 or counts["five_candidate_oracle_090"] != 285:
        raise ValueError("candidate ceiling does not reproduce H0")
    output = root / "v17sel_a0_candidate_ceiling_audit"
    output.mkdir(exist_ok=True)
    write_jsonl(output / "internal300_categories.jsonl", rows)
    summary = {
        "decision": "CANDIDATE_CEILING_DIAGNOSIS_ONLY",
        "train2863_descriptive": dict(train_counts),
        "internal300_design_exposed": dict(counts),
        "development_used": False,
        "confirmation_used": False,
        "artifacts": {"config_sha256": sha256(config), "train_candidates_sha256": sha256(train_path), "internal_candidates_sha256": sha256(internal_path), "selection_sha256": sha256(selection_path)},
    }
    write_metadata(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
