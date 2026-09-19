from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.data.schemas import read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.training.train_v17h0_multianchor_cutoff import meets_fidelity, project


WIDTH = 12
SIZE = 1 << WIDTH


def exact_success(path: Path) -> list[bool]:
    values = [False] * SIZE
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            mask = state_to_mask(row["state"])
            if mask in seen:
                raise ValueError(f"duplicate exact mask {path}: {mask}")
            seen.add(mask)
            values[mask] = meets_fidelity(float(row["fidelity"]), 0.9)
    if len(seen) != SIZE:
        raise ValueError(f"incomplete exact lattice {path}: {len(seen)}")
    return values


def successful_supermask_reachability(success: list[bool]) -> list[bool]:
    reach = success.copy()
    for bit in range(WIDTH):
        flag = 1 << bit
        for mask in range(SIZE):
            if not mask & flag:
                reach[mask] |= reach[mask | flag]
    return reach


def prefixes(order: list[int]) -> list[int]:
    result = [0]
    for packet in order:
        result.append(result[-1] | (1 << packet))
    if len(result) != WIDTH + 1 or result[-1] != SIZE - 1:
        raise ValueError("invalid projected order")
    return result


def audit_query(row: dict, base_order: list[int], exact_dir: Path) -> dict:
    success = exact_success(exact_dir / f"{row['example_id']}.jsonl")
    reach = successful_supermask_reachability(success)
    candidates = row["candidates"]
    if len(candidates) != 5:
        raise ValueError("expected V8 fallback and four V13 mask projections")
    orders = [base_order] + [project(base_order, int(item["mask"])) for item in candidates[1:]]
    paths = [prefixes(order) for order in orders]
    observed = [any(success[mask] for mask in path) for path in paths]
    if observed != [bool(item["success"][3]) for item in candidates]:
        raise ValueError(f"candidate oracle mismatch {row['example_id']}")
    count_by_depth = []
    covered_by_depth = []
    success_seen = False
    first_loss = None
    for depth in range(WIDTH + 1):
        success_seen |= any(success[path[depth]] for path in paths)
        viable_masks = {path[depth] for path in paths if reach[path[depth]]}
        count_by_depth.append(len(viable_masks))
        covered_by_depth.append(bool(success_seen or viable_masks))
        if first_loss is None and not success_seen and not viable_masks:
            first_loss = depth
    chosen_parent_masks = []
    viable_alternatives = []
    chosen_next_actions = []
    if first_loss is not None and first_loss > 0:
        parent_depth = first_loss - 1
        chosen_parent_masks = sorted({path[parent_depth] for path in paths if reach[path[parent_depth]]})
        viable_alternatives = sorted({
            bit for parent in chosen_parent_masks for bit in range(WIDTH)
            if not parent & (1 << bit) and reach[parent | (1 << bit)]
        })
        chosen_next_actions = sorted({order[parent_depth] for order in orders})
    return {
        "example_id": row["example_id"],
        "global_exact_success": any(success),
        "candidate_pool_success": any(observed),
        "candidate_success": observed,
        "unique_viable_prefixes_by_depth": count_by_depth,
        "success_or_viable_coverage_by_depth": covered_by_depth,
        "first_loss_depth": first_loss,
        "viable_parent_masks_before_loss": chosen_parent_masks,
        "viable_alternative_next_actions": viable_alternatives,
        "chosen_next_actions": chosen_next_actions,
        "successful_state_count": sum(success),
    }


def audit_role(candidate_path: Path, orders: dict[str, list[int]], exact_dir: Path) -> tuple[dict, list[dict]]:
    rows = []
    for index, candidate in enumerate(read_jsonl(candidate_path), start=1):
        query = candidate["example_id"]
        rows.append(audit_query(candidate, orders[query], exact_dir))
        if index % 500 == 0:
            print(json.dumps({"role": candidate_path.name, "audited": index}), flush=True)
    pool_misses = [row for row in rows if row["global_exact_success"] and not row["candidate_pool_success"]]
    first_loss = Counter(row["first_loss_depth"] for row in pool_misses)
    if any(value is None for value in first_loss):
        raise ValueError("pool miss lacked a first loss")
    role = {
        "examples": len(rows),
        "global_exact_success": sum(row["global_exact_success"] for row in rows),
        "candidate_pool_success": sum(row["candidate_pool_success"] for row in rows),
        "global_reachable_pool_misses": len(pool_misses),
        "first_loss_depth_histogram_pool_misses": {str(key): value for key, value in sorted(first_loss.items())},
        "alive_proposal_fraction_by_depth_all_global_reachable": [
            sum(row["success_or_viable_coverage_by_depth"][depth] for row in rows if row["global_exact_success"])
            / max(1, sum(row["global_exact_success"] for row in rows))
            for depth in range(WIDTH + 1)
        ],
        "single_viable_parent_at_first_loss": sum(len(row["viable_parent_masks_before_loss"]) == 1 for row in pool_misses),
        "multiple_viable_parents_at_first_loss": sum(len(row["viable_parent_masks_before_loss"]) > 1 for row in pool_misses),
    }
    return role, rows


def main() -> None:
    root = Path("results/v2_rank_then_cut")
    config = Path("configs/v17sel_b0_projection_coverage_audit.json")
    if json.loads(config.read_text())["status"] != "FROZEN_READ_ONLY":
        raise ValueError("unfrozen protocol")
    train_path = root / "v14_train2863_top4_candidates.jsonl"
    internal_path = root / "v14_internal_validation300_top4_candidates.jsonl"
    rollouts = root / "v9_v8_train3163_beam8_rollouts.jsonl"
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(rollouts)}
    exact_dir = root / "candidates5000_exact"
    train_summary, train_rows = audit_role(train_path, orders, exact_dir)
    internal_summary, internal_rows = audit_role(internal_path, orders, exact_dir)
    if train_summary["examples"] != 2863 or internal_summary["examples"] != 300:
        raise ValueError("unexpected split size")
    if internal_summary["candidate_pool_success"] != 285 or internal_summary["global_reachable_pool_misses"] != 15:
        raise ValueError("SEL-A0 ceiling not reproduced")
    output = root / "v17sel_b0_projection_coverage_audit"
    output.mkdir(exist_ok=True)
    write_jsonl(output / "train_pool_misses.jsonl", [row for row in train_rows if row["global_exact_success"] and not row["candidate_pool_success"]])
    write_jsonl(output / "internal_pool_misses.jsonl", [row for row in internal_rows if row["global_exact_success"] and not row["candidate_pool_success"]])
    summary = {
        "decision": "PROJECTION_COVERAGE_DIAGNOSIS_ONLY",
        "train2863": train_summary,
        "internal300_design_exposed": internal_summary,
        "development_used": False,
        "confirmation_used": False,
        "artifacts": {"config_sha256": sha256(config), "train_candidates_sha256": sha256(train_path), "internal_candidates_sha256": sha256(internal_path), "rollouts_sha256": sha256(rollouts)},
    }
    write_metadata(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
