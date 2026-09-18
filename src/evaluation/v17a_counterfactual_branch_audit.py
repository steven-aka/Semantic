from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.search.sequential_trajectory_dp import SequentialTrajectoryDP


def read_details(path: str) -> list[dict[str, Any]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mask_reached(dp: SequentialTrajectoryDP, history: Sequence[int]) -> tuple[int, int]:
    mask, reached = 0, dp.attained[0]
    for packet in history:
        mask |= 1 << int(packet); reached = max(reached, dp.attained[mask])
    return mask, reached


def superset_minimum_tokens(exact: Sequence[ExactSearchResult], levels: Sequence[float]) -> list[list[float]]:
    width = len(exact[0].state); size = 1 << width
    tokens = [0] * size; fidelity = [0.0] * size
    for row in exact:
        mask = state_to_mask(row.state); tokens[mask] = int(row.tokens); fidelity[mask] = float(row.fidelity)
    output = []
    for level in levels:
        values = [float(tokens[mask]) if fidelity[mask] + 1e-12 >= level else math.inf for mask in range(size)]
        for bit in range(width):
            flag = 1 << bit
            for mask in range(size):
                if not mask & flag:
                    values[mask] = min(values[mask], values[mask | flag])
        output.append(values)
    return output


def state_consequences(dp: SequentialTrajectoryDP, minimum_tokens: Sequence[Sequence[float]], history: Sequence[int], full_tokens: int, *, cost_gap: float) -> dict[str, Any] | None:
    mask, reached = mask_reached(dp, history)
    if mask == dp.size - 1: return None
    actions = []
    for packet, value in dp.action_values(mask, reached):
        next_mask = mask | (1 << packet); next_reached = max(reached, dp.attained[next_mask])
        q = []
        for index in range(len(dp.levels)):
            if index < next_reached: q.append((1, 0.0))
            else:
                token = minimum_tokens[index][next_mask]
                q.append((int(math.isfinite(token)), None if not math.isfinite(token) else token / full_tokens))
        actions.append({"packet": packet, "final_reached": value.reached_levels, "normalized_global_cost": value.additional_cumulative_tokens / (len(dp.levels) * full_tokens), "q": q})
    final_counts = [row["final_reached"] for row in actions]
    best_reach = max(final_counts)
    primary_index = next((i for i, level in enumerate(dp.levels) if abs(level - .9) < 1e-9), None)
    primary_flags = [row["q"][primary_index][0] for row in actions] if primary_index is not None else []
    equal_reach_costs = [row["normalized_global_cost"] for row in actions if row["final_reached"] == best_reach]
    feasibility_critical = min(final_counts) < best_reach
    primary_critical = bool(primary_flags) and min(primary_flags) < max(primary_flags)
    cost_critical = not feasibility_critical and max(equal_reach_costs) - min(equal_reach_costs) >= cost_gap
    # Consequence-only signatures deliberately exclude query type, depth, and packet identity.
    feasibility_signature = tuple(sorted((row["final_reached"], tuple(v for v, _ in row["q"])) for row in actions))
    cost_signature = tuple(sorted((row["final_reached"], tuple(v for v, _ in row["q"]), tuple(99 if c is None else round(c / .05) for _, c in row["q"])) for row in actions))
    return {"history": list(history), "mask": mask, "reached": reached, "actions": actions, "feasibility_critical": feasibility_critical, "primary_090_critical": primary_critical, "cost_critical": cost_critical, "feasibility_signature": repr(feasibility_signature), "cost_signature": repr(cost_signature)}


def worker(task: tuple[dict[str, Any], list[int], dict[str, Any] | None, str, int, float]) -> dict[str, Any]:
    source, order, detail, exact_dir, hops, cost_gap = task
    exact = list(read_jsonl(Path(exact_dir) / f"{source['example_id']}.jsonl", ExactSearchResult))
    dp = SequentialTrajectoryDP(exact, source["active_levels"]); minima = superset_minimum_tokens(exact, dp.levels); full_tokens = dp.tokens[-1]
    histories = [tuple(order[:depth]) for depth in range(12)]
    frontier = list(histories); seen = set(histories); by_hop = []; hop_fs = []; hop_cs = []
    for hop in range(hops + 1):
        records = [state_consequences(dp, minima, history, full_tokens, cost_gap=cost_gap) for history in frontier]
        records = [row for row in records if row is not None]
        critical_records = [row for row in records if row["feasibility_critical"] or row["cost_critical"]]
        hop_fs.append(sorted({row["feasibility_signature"] for row in critical_records}))
        hop_cs.append(sorted({row["cost_signature"] for row in critical_records}))
        by_hop.append({"states": len(records), "feasibility_critical": sum(row["feasibility_critical"] for row in records), "primary_090_critical": sum(row["primary_090_critical"] for row in records), "cost_critical": sum(row["cost_critical"] for row in records), "action_examples": sum(len(row["actions"]) for row in records if row["feasibility_critical"] or row["cost_critical"])})
        next_frontier = []
        if hop < hops:
            for history in frontier:
                used = set(history)
                for packet in range(12):
                    child = history + (packet,)
                    if packet not in used and child not in seen and len(child) < 12:
                        seen.add(child); next_frontier.append(child)
        frontier = next_frontier
    deployed = [state_consequences(dp, minima, history, full_tokens, cost_gap=cost_gap) for history in histories]
    deployed = [row for row in deployed if row is not None]
    critical = [row for row in deployed if row["feasibility_critical"] or row["cost_critical"]]
    fid = None
    if detail is not None and detail.get("first_primary_090_extinction_depth") is not None:
        depth = int(detail["first_primary_090_extinction_depth"])
        parents = [()] if depth == 1 else [tuple(x) for x in detail["trace"][depth - 2]["histories"]]
        values = [state_consequences(dp, minima, history, full_tokens, cost_gap=cost_gap) for history in parents]
        values = [row for row in values if row is not None]
        fid = {"parents": len(values), "feasibility_critical_parents": sum(row["feasibility_critical"] for row in values), "primary_critical_parents": sum(row["primary_090_critical"] for row in values), "cost_critical_parents": sum(row["cost_critical"] for row in values), "any_critical": any(row["feasibility_critical"] or row["cost_critical"] for row in values), "feasibility_signatures": [row["feasibility_signature"] for row in values if row["feasibility_critical"] or row["cost_critical"]], "cost_signatures": [row["cost_signature"] for row in values if row["feasibility_critical"] or row["cost_critical"]]}
    return {"example_id": source["example_id"], "by_hop": by_hop, "hop_feasibility_signatures": hop_fs, "hop_cost_signatures": hop_cs, "critical_states": [{key: row[key] for key in ("history", "mask", "reached", "feasibility_critical", "primary_090_critical", "cost_critical", "feasibility_signature", "cost_signature")} for row in critical], "fid": fid}


def aggregate(rows: Sequence[dict[str, Any]], train_feasibility: Sequence[set[str]] | None = None, train_cost: Sequence[set[str]] | None = None) -> dict[str, Any]:
    hops = len(rows[0]["by_hop"]); by_hop = []
    for hop in range(hops):
        by_hop.append({key: sum(row["by_hop"][hop][key] for row in rows) for key in rows[0]["by_hop"][hop]})
    fids = [row["fid"] for row in rows if row["fid"] is not None]
    result = {"examples": len(rows), "by_hop": by_hop, "fid_queries": len(fids), "fid_queries_with_any_critical_parent": sum(row["any_critical"] for row in fids), "fid_parents": sum(row["parents"] for row in fids), "fid_feasibility_critical_parents": sum(row["feasibility_critical_parents"] for row in fids), "fid_primary_critical_parents": sum(row["primary_critical_parents"] for row in fids), "fid_cost_critical_parents": sum(row["cost_critical_parents"] for row in fids)}
    if train_feasibility is not None and train_cost is not None:
        fs = [signature for row in fids for signature in row["feasibility_signatures"]]; cs = [signature for row in fids for signature in row["cost_signatures"]]
        result["fid_q_structure_coverage"] = {
            f"through_hop_{hop}": {
                "critical_parent_feasibility_signature_exact": sum(x in train_feasibility[hop] for x in fs),
                "critical_parent_feasibility_signature_total": len(fs),
                "critical_parent_cost_signature_exact": sum(x in train_cost[hop] for x in cs),
                "critical_parent_cost_signature_total": len(cs),
            }
            for hop in range(len(train_feasibility))
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="V17-A counterfactual decision-critical branch audit")
    parser.add_argument("--train-data", required=True); parser.add_argument("--validation-data", required=True); parser.add_argument("--development-data", required=True)
    parser.add_argument("--rollouts", required=True); parser.add_argument("--development-orders", required=True); parser.add_argument("--train-details", required=True); parser.add_argument("--validation-details", required=True); parser.add_argument("--development-details", required=True)
    parser.add_argument("--exact-dir", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--workers", type=int, default=8); parser.add_argument("--hops", type=int, default=2); parser.add_argument("--cost-gap", type=float, default=.03)
    args = parser.parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    all_orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    dev_orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.development_orders)}
    role_specs = [("train2863", args.train_data, args.train_details, all_orders), ("internal_validation300", args.validation_data, args.validation_details, all_orders), ("consumed_development300", args.development_data, args.development_details, dev_orders)]
    role_rows = {}
    with Pool(args.workers) as pool:
        for role, data_path, details_path, orders in role_specs:
            data = list(read_jsonl(data_path)); details = {row["example_id"]: row for row in read_details(details_path)}
            tasks = [
                ({"example_id": row["example_id"], "active_levels": row.get("attainable_levels", row["active_levels"])}, orders[row["example_id"]], details.get(row["example_id"]), args.exact_dir, args.hops, args.cost_gap)
                for row in data
            ]
            values = list(pool.imap(worker, tasks, chunksize=4)); role_rows[role] = values
            compact = [{key: value for key, value in row.items() if not key.startswith("hop_")} for row in values]
            write_jsonl(out / f"{role}_audit.jsonl", compact)
    train_fs, train_cs = [], []
    cumulative_fs: set[str] = set(); cumulative_cs: set[str] = set()
    for hop in range(args.hops + 1):
        cumulative_fs |= {signature for row in role_rows["train2863"] for signature in row["hop_feasibility_signatures"][hop]}
        cumulative_cs |= {signature for row in role_rows["train2863"] for signature in row["hop_cost_signatures"][hop]}
        train_fs.append(set(cumulative_fs)); train_cs.append(set(cumulative_cs))
    summary = {"complete": True, "critical_rule": {"feasibility": "legal next actions differ in maximum eventually reachable active-anchor count", "primary_090": "legal next actions differ in whether fidelity 0.90 remains reachable", "cost": f"all actions preserve maximum reach but normalized exact-DP future-cost range >= {args.cost_gap}"}, "state": "query plus ordered history, selected mask, and reached-anchor count", "roles": {role: aggregate(values, None if role == "train2863" else train_fs, None if role == "train2863" else train_cs) for role, values in role_rows.items()}, "train_unique_q_signatures": {f"through_hop_{hop}": {"feasibility": len(train_fs[hop]), "cost_binned_005": len(train_cs[hop])} for hop in range(args.hops + 1)}, "restrictions": {"train_role_examples": 2863, "internal_validation_not_training": True, "development_consumed_diagnostic_only": True, "locked_roles_used": False}, "artifacts_sha256": {path: sha256(path) for path in (args.train_data, args.validation_data, args.development_data, args.rollouts, args.development_orders, args.train_details, args.validation_details, args.development_details)}}
    write_metadata(out / "summary.json", summary); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
