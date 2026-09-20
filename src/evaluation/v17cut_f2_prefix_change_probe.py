"""Grouped prefix-change probe for conservative lower-anchor early exits."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.v17cut_f1_conservative_early_exit_risk import evaluate, load_outcomes
from src.evaluation.v17dep_a0_selector_probe import average_precision, train_linear
from src.model.mask_value_head import MaskValueHead
from src.reproducibility import sha256, write_metadata
from src.training.train_v17cut_b0_structured_fixed_v8 import prefix_masks


def mask_logits(rows: list[dict], cache: dict, orders: dict[str, list[int]],
                head: MaskValueHead, device: torch.device) -> tuple[list[float], list[float]]:
    positions = {query: index for index, query in enumerate(cache["example_ids"])}
    prefixes = [prefix_masks(orders[row["example_id"]]) for row in rows]
    out8, out9 = [], []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for start in range(0, len(rows), 64):
            subset = rows[start:start + 64]
            indices = torch.tensor([positions[row["example_id"]] for row in subset])
            packets = cache["packets"][indices].float().to(device)
            questions = cache["questions"][indices].float().to(device)
            masks = torch.tensor([[prefixes[start + offset][8], prefixes[start + offset][9]]
                                  for offset in range(len(subset))], dtype=torch.long, device=device)
            logits = head(packets, questions, masks.flatten(),
                          torch.arange(len(subset), device=device).repeat_interleave(2))
            values = logits[:, 2].float().reshape(len(subset), 2).cpu().tolist()
            out8.extend(value[0] for value in values)
            out9.extend(value[1] for value in values)
    return out8, out9


def curve(rows: list[dict], scores: list[float]) -> list[dict]:
    by_rank = sorted(scores, reverse=True)
    result = []
    for fraction in (.01, .02, .05, .10, .20):
        threshold = by_rank[max(1, math.floor(len(rows) * fraction)) - 1]
        records = [{**row, "score": score} for row, score in zip(rows, scores)]
        result.append({"nominal_fraction": fraction, "threshold": threshold,
                       **evaluate(records, threshold)})
    return result


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
    if config["status"] != "FROZEN_TRAIN_ONLY_GROUPED_OOF":
        raise ValueError("unfrozen grouped probe")
    rows = [row for row in read_jsonl(args.candidates) if fold(row["example_id"]) != 4]
    if len(rows) != 1421:
        raise ValueError("unexpected train-side count")
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    head = MaskValueHead(model_dim=512, layers=2, heads=8, dropout=.1)
    head.load_state_dict(torch.load(args.head, map_location="cpu", weights_only=True))
    head.to(device).eval()
    z8, z9 = mask_logits(rows, cache, orders, head, device)
    outcomes = load_outcomes(rows, orders, Path(args.exact_dir), z9)
    unsafe = torch.tensor([any(row["base"][i] is True and row["early"][i] is False for i in range(3))
                           for row in outcomes], dtype=torch.bool)
    features = torch.tensor([[a, b, b - a, row["normalized_saving"]]
                             for a, b, row in zip(z8, z9, outcomes)], dtype=torch.float32)
    folds = [fold(row["example_id"]) for row in rows]
    oof_risk = torch.empty(len(rows), dtype=torch.float32)
    torch.manual_seed(20260920)
    settings = {"epochs": 100, "batch_size": 512, "learning_rate": .01,
                "weight_decay": .001, "seed": 20260920}
    for heldout in range(4):
        train = torch.tensor([index for index, value in enumerate(folds) if value != heldout])
        test = torch.tensor([index for index, value in enumerate(folds) if value == heldout])
        mu, sigma, model = train_linear(features[train], unsafe[train], settings, device)
        with torch.no_grad():
            oof_risk[test] = model(((features[test] - mu) / sigma).to(device)).squeeze(-1).cpu()
    safety = (-oof_risk).tolist()
    dynamic_curve = curve(outcomes, safety)
    static_curve = curve(outcomes, z9)
    candidate = next(item for item in dynamic_curve if item["nominal_fraction"] == .10)
    passed = (candidate["switches"] >= 100 and candidate["complete_breaks"] == 0
              and not any(candidate["anchor_breaks"])
              and candidate["baseline_mean_final_context_fraction"] -
              candidate["policy_mean_final_context_fraction"] >= .005)
    result = {
        "protocol": config["protocol"], "unsafe_prevalence": float(unsafe.float().mean()),
        "unsafe_average_precision": average_precision(list(zip(oof_risk.tolist(), unsafe.tolist()))),
        "static_v13_curve": static_curve, "sequential_grouped_oof_curve": dynamic_curve,
        "train_opening_gate_passed": passed,
        "fold4_611_read": False, "holdout_581_read": False,
        "internal300_read": False, "development_read": False,
        "confirmation_read": False, "new_target_calls": 0,
        "artifacts": {"config_sha256": sha256(args.config), "candidates_sha256": sha256(args.candidates),
                      "cache_sha256": sha256(args.cache), "head_sha256": sha256(args.head),
                      "rollouts_sha256": sha256(args.rollouts)},
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train1421_oof.jsonl", [
        {"example_id": row["example_id"], "z080_depth8": z8[index],
         "z080_depth9": z9[index], "normalized_saving": row["normalized_saving"],
         "unsafe": bool(unsafe[index]), "oof_risk_logit": float(oof_risk[index])}
        for index, row in enumerate(outcomes)])
    write_metadata(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
