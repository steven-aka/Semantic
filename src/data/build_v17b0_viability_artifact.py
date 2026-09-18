from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Sequence

from src.data.schemas import ExactSearchResult, read_jsonl
from src.evaluation.v16c0_first_irreversible_divergence import trajectory_levels
from src.evaluation.v17a_counterfactual_branch_audit import mask_reached
from src.reproducibility import sha256, write_metadata
from src.search.sequential_trajectory_dp import SequentialTrajectoryDP
from src.training.v17b_viability import FIDELITY_GRID


def build_state(dp: SequentialTrajectoryDP, history: Sequence[int], attainable: Sequence[float], provenance: str) -> dict[str, Any] | None:
    mask, reached = mask_reached(dp, history)
    if mask == dp.size - 1:
        return None
    grid_to_local = {round(level, 2): index for index, level in enumerate(attainable)}
    statuses = []
    for level in FIDELITY_GRID:
        local = grid_to_local.get(round(level, 2))
        statuses.append("undefined" if local is None else ("resolved" if local < reached else "active"))
    actions = []
    for packet, value in dp.action_values(mask, reached):
        labels = []
        for level, status in zip(FIDELITY_GRID, statuses):
            if status != "active":
                labels.append(None)
            else:
                local = grid_to_local[round(level, 2)]
                labels.append(int(value.reached_levels >= local + 1))
        actions.append({"packet": packet, "future_viability": labels})
    active_columns = [i for i, status in enumerate(statuses) if status == "active"]
    critical_levels = [
        FIDELITY_GRID[i] for i in active_columns
        if len({action["future_viability"][i] for action in actions}) > 1
    ]
    if not critical_levels:
        return None
    optimal = list(dp.optimal_actions(mask, reached))
    return {
        "history": list(history), "selected_mask": mask, "depth": len(history),
        "reached_level_count": reached, "anchor_status": statuses,
        "active_target_mask": [status == "active" for status in statuses],
        "critical_levels": critical_levels, "actions": actions,
        "exact_dp_optimal_actions": optimal, "provenance": provenance,
    }


def worker(task: tuple[dict[str, Any], list[int], str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source, order, exact_dir = task
    exact = list(read_jsonl(Path(exact_dir) / f"{source['example_id']}.jsonl", ExactSearchResult))
    attainable = trajectory_levels(source)
    dp = SequentialTrajectoryDP(exact, attainable)
    deployed_histories = [tuple(order[:depth]) for depth in range(12)]
    seen = set(deployed_histories)
    one_hop_histories = []
    for history in deployed_histories:
        used = set(history)
        for packet in range(12):
            child = history + (packet,)
            if packet not in used and len(child) < 12 and child not in seen:
                seen.add(child); one_hop_histories.append(child)
    rows = []
    for provenance, histories in (("deployed", deployed_histories), ("one_hop", one_hop_histories)):
        for history in histories:
            row = build_state(dp, history, attainable, provenance)
            if row is not None:
                rows.append({"example_id": source["example_id"], **row})
    return rows, {"anchor_count": len(attainable), "deployed_scanned": len(deployed_histories), "one_hop_scanned": len(one_hop_histories)}


def write_gzip_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audited V17-B0 multi-anchor viability artifact")
    parser.add_argument("--train-data", required=True); parser.add_argument("--rollouts", required=True)
    parser.add_argument("--exact-dir", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    sources = list(read_jsonl(args.train_data)); orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    tasks = [(row, orders[row["example_id"]], args.exact_dir) for row in sources]
    all_rows = []; audits = []
    with Pool(args.workers) as pool:
        for rows, audit in pool.imap(worker, tasks, chunksize=4):
            all_rows.extend(rows); audits.append(audit)
    keys = [(row["example_id"], tuple(row["history"])) for row in all_rows]
    if len(keys) != len(set(keys)): raise RuntimeError("canonical state key is not unique")
    counts = Counter(row["provenance"] for row in all_rows)
    anchor_counts = Counter(row["anchor_count"] for row in audits)
    if anchor_counts != Counter({5: 2248, 4: 615}): raise RuntimeError(f"unexpected train anchor contracts: {anchor_counts}")
    if counts != Counter({"deployed": 6200, "one_hop": 41118}): raise RuntimeError(f"V17-A critical counts not reproduced: {counts}")
    four_anchor = {row["example_id"] for row in sources if len(trajectory_levels(row)) == 4}
    if any(row["anchor_status"][4] != "undefined" or row["active_target_mask"][4] for row in all_rows if row["example_id"] in four_anchor):
        raise RuntimeError("four-anchor example emitted an active 0.95 label")
    artifact = out / "train2863_deployed_hop1_viability_states.jsonl.gz"; write_gzip_jsonl(artifact, all_rows)
    audit = {
        "complete": True, "examples": len(sources), "canonical_states": len(all_rows),
        "states_by_provenance": dict(counts), "contract_examples": {"five_anchor": 2248, "four_anchor": 615},
        "four_anchor_095": "undefined_and_masked", "dedup_key": ["example_id", "ordered_history"],
        "state_row_schema": "actions nested within each state; no action-row expansion",
        "target_semantics": {"undefined": "anchor absent from attainable_levels; never a negative", "resolved": "anchor already attained; excluded from main loss", "active": "exact future reachability after action"},
        "inputs": {"train_data": args.train_data, "train_data_sha256": sha256(args.train_data), "rollouts": args.rollouts, "rollouts_sha256": sha256(args.rollouts), "exact_dir": args.exact_dir},
        "artifact": str(artifact), "artifact_sha256": sha256(artifact), "locked_roles_used": False,
    }
    write_metadata(out / "artifact_audit.json", audit); print(json.dumps(audit, indent=2))


if __name__ == "__main__": main()
