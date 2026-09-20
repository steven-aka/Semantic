"""Train-only risk--coverage curve, then one frozen-threshold exposed replay."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import torch

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl, write_jsonl
from src.model.mask_value_head import MaskValueHead
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.training.train_v17cut_b0_structured_fixed_v8 import prefix_masks
from src.training.train_v17h0_multianchor_cutoff import LEVELS, meets_fidelity


def get_scores(rows: list[dict], positions: dict[str, int], cache: dict,
               orders: dict[str, list[int]], head: MaskValueHead,
               device: torch.device) -> list[float]:
    masks = torch.tensor([prefix_masks(orders[row["example_id"]])[9] for row in rows], dtype=torch.long)
    scores = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for start in range(0, len(rows), 64):
            selected = rows[start:start + 64]
            ids = torch.tensor([positions[row["example_id"]] for row in selected])
            result = head(cache["packets"][ids].float().to(device),
                          cache["questions"][ids].float().to(device),
                          masks[start:start + len(selected)].to(device),
                          torch.arange(len(selected), device=device))
            scores.extend(result[:, 2].float().cpu().tolist())
    return scores


def load_outcomes(rows: list[dict], orders: dict[str, list[int]], exact_dir: Path,
                  scores: list[float]) -> list[dict]:
    answer = []
    for number, (row, score) in enumerate(zip(rows, scores), 1):
        query = row["example_id"]
        prefixes = prefix_masks(orders[query])
        desired = {prefixes[9], prefixes[10]}
        values = {}
        for record in read_jsonl(exact_dir / f"{query}.jsonl"):
            mask = state_to_mask(record["state"])
            if mask in desired:
                values[mask] = record
        if set(values) != desired:
            raise ValueError(f"missing prefixes: {query}")
        active = [level in row["attainable_levels"] for level in LEVELS]
        fidelity9 = values[prefixes[9]]["fidelity"]
        fidelity10 = values[prefixes[10]]["fidelity"]
        base = [meets_fidelity(fidelity10, level) if active[i] else None for i, level in enumerate(LEVELS)]
        early = [meets_fidelity(fidelity9 if i < 3 else fidelity10, level) if active[i] else None
                 for i, level in enumerate(LEVELS)]
        saving = 3 * (values[prefixes[10]]["tokens"] - values[prefixes[9]]["tokens"]) / (sum(active) * row["full_tokens"])
        answer.append({"example_id": query, "score": score, "base": base, "early": early,
                       "context10": values[prefixes[10]]["tokens"] / row["full_tokens"],
                       "normalized_saving": saving})
        if number % 500 == 0:
            print(json.dumps({"loaded": number, "total": len(rows)}), flush=True)
    return answer


def evaluate(rows: list[dict], threshold: float) -> dict:
    switched = [row for row in rows if row["score"] >= threshold]
    breaks = [sum(row["base"][i] is True and row["early"][i] is False for row in switched) for i in range(5)]
    gains = [sum(row["base"][i] is False and row["early"][i] is True for row in switched) for i in range(5)]
    complete_break = sum(all(value is not False for value in row["base"]) and
                         not all(value is not False for value in row["early"]) for row in switched)
    complete_gain = sum(not all(value is not False for value in row["base"]) and
                        all(value is not False for value in row["early"]) for row in switched)
    return {
        "queries": len(rows), "switches": len(switched), "coverage": len(switched) / len(rows),
        "anchor_breaks": breaks, "anchor_gains": gains,
        "complete_breaks": complete_break, "complete_gains": complete_gain,
        "baseline_090": sum(row["base"][3] is True for row in rows),
        "policy_090": sum(row["base"][3] is True for row in rows) - breaks[3] + gains[3],
        "baseline_complete": sum(all(value is not False for value in row["base"]) for row in rows),
        "policy_complete": sum(all(value is not False for value in row["base"]) for row in rows) - complete_break + complete_gain,
        "baseline_mean_final_context_fraction": mean(row["context10"] for row in rows),
        "policy_mean_final_context_fraction": mean(row["context10"] - (row["normalized_saving"] if row["score"] >= threshold else 0.0) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_TRAIN_THEN_ONE_EXPOSED_ROLE":
        raise ValueError("unfrozen risk protocol")
    source = list(read_jsonl(args.candidates))
    train = [row for row in source if fold(row["example_id"]) != 4]
    exposed = [row for row in source if fold(row["example_id"]) == 4]
    if len(train) != 1421 or len(exposed) != 611:
        raise ValueError("unexpected role split")
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    positions = {query: index for index, query in enumerate(cache["example_ids"])}
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    head = MaskValueHead(model_dim=512, layers=2, heads=8, dropout=.1)
    head.load_state_dict(torch.load(args.head, map_location="cpu", weights_only=True))
    head.to(device).eval()
    train_scores = get_scores(train, positions, cache, orders, head, device)
    train_data = load_outcomes(train, orders, Path(args.exact_dir), train_scores)
    descending = sorted(train_scores, reverse=True)
    curve = []
    for fraction in (.01, .02, .05, .10, .20):
        index = max(1, math.floor(len(train) * fraction)) - 1
        threshold = descending[index]
        curve.append({"nominal_fraction": fraction, "threshold": threshold,
                      **evaluate(train_data, threshold)})
    selected = next(row for row in curve if row["nominal_fraction"] == .10)
    gate = (selected["switches"] >= 100 and selected["complete_breaks"] == 0
            and not any(selected["anchor_breaks"])
            and selected["policy_mean_final_context_fraction"] < selected["baseline_mean_final_context_fraction"])
    result = {
        "protocol": config["protocol"], "train_risk_curve": curve,
        "frozen_threshold": selected["threshold"], "train_opening_gate_passed": gate,
        "fold4_611_read": False, "holdout_581_read": False,
        "internal300_read": False, "development_read": False,
        "confirmation_read": False, "new_target_calls": 0,
        "artifacts": {"config_sha256": sha256(args.config), "candidates_sha256": sha256(args.candidates),
                      "cache_sha256": sha256(args.cache), "head_sha256": sha256(args.head),
                      "rollouts_sha256": sha256(args.rollouts)},
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if gate:
        exposed_scores = get_scores(exposed, positions, cache, orders, head, device)
        exposed_data = load_outcomes(exposed, orders, Path(args.exact_dir), exposed_scores)
        result["design_exposed_611"] = evaluate(exposed_data, selected["threshold"])
        result["fold4_611_read"] = True
        write_jsonl(output / "exposed_611_per_query.jsonl", exposed_data)
    write_jsonl(output / "train1421_per_query.jsonl", train_data)
    write_metadata(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
