from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from src.data.schemas import ExactSearchResult, read_jsonl
from src.reproducibility import sha256, write_metadata
from src.search.rank_then_cut import best_prefix_nested_chain


def read_gzip(path: str) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def contract_table(data: str, exact_dir: str, baseline: list[dict], candidate: list[dict]) -> dict:
    rows = {row["example_id"]: row for row in read_jsonl(data)}
    orders = {
        "baseline": {row["example_id"]: row["decoded_order"] for row in baseline},
        "candidate": {row["example_id"]: row["decoded_order"] for row in candidate},
    }
    counts = {label: {} for label in orders}; complete = {label: 0 for label in orders}; success_090 = {}
    for example_id, source in rows.items():
        exact = list(read_jsonl(Path(exact_dir) / f"{example_id}.jsonl", ExactSearchResult))
        for label in orders:
            predictions = best_prefix_nested_chain(exact, orders[label][example_id], source["active_levels"])
            all_success = True
            for level, prediction in zip(source["active_levels"], predictions):
                key = str(float(level)); passed = bool(prediction["feasible"])
                counts[label][key] = counts[label].get(key, 0) + int(passed); all_success &= passed
                if abs(float(level) - .9) < 1e-9: success_090[label, example_id] = passed
            complete[label] += int(all_success)
    repairs = sum(not success_090["baseline", key] and success_090["candidate", key] for key in rows)
    breaks = sum(success_090["baseline", key] and not success_090["candidate", key] for key in rows)
    return {"per_level": counts, "complete": complete, "fidelity_090_repairs": repairs, "fidelity_090_breaks": breaks, "fidelity_090_net": repairs - breaks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-data", required=True); parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--validation-baseline-gzip", required=True); parser.add_argument("--validation-candidate", required=True)
    parser.add_argument("--validation-fid-summary", required=True); parser.add_argument("--development-baseline", required=True)
    parser.add_argument("--development-candidate", required=True); parser.add_argument("--development-summary", required=True)
    parser.add_argument("--development-v8-fid-summary", required=True); parser.add_argument("--development-v16c1-fid-summary", required=True)
    parser.add_argument("--output", required=True); args = parser.parse_args()
    internal = contract_table(args.validation_data, args.exact_dir, read_gzip(args.validation_baseline_gzip), list(read_jsonl(args.validation_candidate)))
    baseline_dev = {row["example_id"]: row for row in read_jsonl(args.development_baseline)}
    candidate_dev = {row["example_id"]: row for row in read_jsonl(args.development_candidate)}
    repairs = breaks = 0
    for key in baseline_dev:
        old = next(row for row in baseline_dev[key]["anchors"] if abs(row["fidelity_level"] - .9) < 1e-9)["contract_success"]
        new = next(row for row in candidate_dev[key]["anchors"] if abs(row["fidelity_level"] - .9) < 1e-9)["contract_success"]
        repairs += int(not old and new); breaks += int(old and not new)
    development = json.loads(Path(args.development_summary).read_text())
    v8_fid = json.loads(Path(args.development_v8_fid_summary).read_text())
    c1_fid = json.loads(Path(args.development_v16c1_fid_summary).read_text())
    result = {
        "decision": "STOP_STATIC_FID_CORRECTION",
        "internal_validation_gate_passed": True,
        "development_gate_passed": False,
        "internal_validation": internal,
        "development": development,
        "development_relative_to_v8_0.90": {"repairs": repairs, "breaks": breaks, "net": repairs - breaks},
        "post_stop_mechanism_audit": {
            "v8_primary_090_beam_extinctions": v8_fid["failure_mechanism"]["among_primary_090_failures"]["beam_pruning_extinction"],
            "v16c1_primary_090_beam_extinctions": c1_fid["failure_mechanism"]["among_primary_090_failures"]["beam_pruning_extinction"],
            "interpretation": "The static correction changed only one consumed-development beam-extinction outcome and did not improve authoritative 0.90 success. Internal-validation gains did not transfer."
        },
        "second_aggregation_round_opened": False,
        "learned_cutoff_opened": False,
        "fresh_confirmation_opened": False,
        "next_step": "Do not tune the static FID loss. Reassess whether query-specific viable-child supervision can generalize from only 197 train failures, and separate terminal candidate selection from path survival before another model run.",
        "artifacts_sha256": {path: sha256(path) for path in vars(args).values() if isinstance(path, str) and Path(path).is_file()},
    }
    for path in (args.validation_candidate, args.development_candidate):
        compressed = f"{path}.gz"
        if Path(compressed).is_file():
            result["artifacts_sha256"][compressed] = sha256(compressed)
    write_metadata(args.output, result); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
