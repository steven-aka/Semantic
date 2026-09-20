"""Correct F3's size association for depth and prior success eligibility."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.schemas import read_jsonl
from src.reproducibility import sha256, write_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--transitions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "READ_ONLY_CORRECTION_TO_F3":
        raise ValueError("unfrozen correction")
    records = list(read_jsonl(args.transitions))
    if len(records) != 1421 * 12:
        raise ValueError("unexpected F3 transition count")
    queries = sorted({row["example_id"] for row in records})
    if len(queries) != 1421:
        raise ValueError("unexpected F3 query count")
    positions = {query: index for index, query in enumerate(queries)}
    # [query, depth 10..12, size group (other, large), (risk-set count, rollback count)]
    cells = np.zeros((len(queries), 3, 2, 2), dtype=np.int64)
    raw = defaultdict(lambda: [0, 0])
    for row in records:
        size = int(row["large_within_query"])
        raw[size][0] += 1
        raw[size][1] += int(row["rollback_090"])
        depth = int(row["depth_after"])
        if depth in (10, 11, 12) and float(row["fidelity_before"]) + 1e-6 >= .9:
            cell = cells[positions[row["example_id"]], depth - 10, size]
            cell[0] += 1
            cell[1] += int(row["rollback_090"])
    totals = cells.sum(axis=0)
    if np.any(totals[:, :, 0] == 0):
        raise ValueError("empty depth by size risk stratum")
    depth_weights = totals[:, :, 0].sum(axis=1).astype(float)
    depth_weights /= depth_weights.sum()

    def standardized_ratio(value: np.ndarray) -> float:
        rate = value[:, :, 1] / value[:, :, 0]
        standardized = (rate * depth_weights[:, None]).sum(axis=0)
        return float(standardized[1] / standardized[0])

    ratio = standardized_ratio(totals)
    rng = np.random.default_rng(20260920)
    bootstrap = []
    for _ in range(1000):
        indices = rng.integers(0, len(queries), len(queries))
        sample = cells[indices].sum(axis=0)
        if np.any(sample[:, :, 0] == 0) or (sample[:, 0, 1] == 0).all():
            continue
        bootstrap.append(standardized_ratio(sample))
    if len(bootstrap) < 990:
        raise ValueError("unstable bootstrap support")
    per_depth = {}
    for offset, depth in enumerate((10, 11, 12)):
        other, large = totals[offset]
        per_depth[str(depth)] = {
            "other_at_risk": int(other[0]), "other_rollbacks": int(other[1]),
            "other_risk": float(other[1] / other[0]),
            "large_at_risk": int(large[0]), "large_rollbacks": int(large[1]),
            "large_risk": float(large[1] / large[0]),
            "large_to_other_risk_ratio": float((large[1] / large[0]) / (other[1] / other[0])),
        }
    result = {
        "protocol": config["protocol"], "queries": len(queries), "transitions": len(records),
        "unadjusted_large_to_other_rollback_rate_ratio": (raw[1][1] / raw[1][0]) / (raw[0][1] / raw[0][0]),
        "at_risk_depth10_12": per_depth,
        "depth_standardization_weights": {str(depth): float(depth_weights[index]) for index, depth in enumerate((10, 11, 12))},
        "depth_standardized_large_to_other_risk_ratio": ratio,
        "query_cluster_bootstrap_95": [float(np.quantile(bootstrap, .025)), float(np.quantile(bootstrap, .975))],
        "decision": "F3_UNADJUSTED_GATE_CONFOUNDED_NO_SPLIT_PILOT_AUTHORIZATION",
        "fold4_611_read": False, "holdout_581_read": False,
        "internal300_read": False, "development_read": False,
        "confirmation_read": False, "new_target_calls": 0,
        "artifacts": {"config_sha256": sha256(args.config), "transitions_sha256": sha256(args.transitions)},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_metadata(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
