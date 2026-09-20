"""Conservative lower-anchor compression from a frozen mask-value signal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl
from src.model.mask_value_head import MaskValueHead
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.training.train_v17cut_b0_structured_fixed_v8 import prefix_masks
from src.training.train_v17h0_multianchor_cutoff import LEVELS, meets_fidelity


def evaluate(rows: list[dict], orders: dict[str, list[int]], cache: dict,
             head: MaskValueHead, exact_dir: Path, device: torch.device) -> dict:
    positions = {query: index for index, query in enumerate(cache["example_ids"])}
    if len(positions) != len(cache["example_ids"]):
        raise ValueError("duplicate cache IDs")
    prefix = [prefix_masks(orders[row["example_id"]]) for row in rows]
    mask9 = torch.tensor([item[9] for item in prefix], dtype=torch.long)
    logits = torch.empty(len(rows), dtype=torch.float32)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for start in range(0, len(rows), 64):
            subset = rows[start:start + 64]
            indices = torch.tensor([positions[row["example_id"]] for row in subset])
            packet = cache["packets"][indices].float().to(device)
            question = cache["questions"][indices].float().to(device)
            output = head(packet, question, mask9[start:start + len(subset)].to(device),
                          torch.arange(len(subset), device=device))
            logits[start:start + len(subset)] = output[:, 2].float().cpu()
    table = []
    for number, row in enumerate(rows, 1):
        query = row["example_id"]
        desired = {prefix[number - 1][9], prefix[number - 1][10]}
        exact = {}
        for record in read_jsonl(exact_dir / f"{query}.jsonl"):
            mask = state_to_mask(record["state"])
            if mask in desired:
                exact[mask] = record
        if set(exact) != desired:
            raise ValueError(f"missing prefix: {query}")
        selected = logits[number - 1].item() >= 4.0
        active = [level in row["attainable_levels"] for level in LEVELS]
        baseline_flags = [meets_fidelity(exact[prefix[number - 1][10]]["fidelity"], level) if active[i] else None
                          for i, level in enumerate(LEVELS)]
        policy_flags = [meets_fidelity(exact[prefix[number - 1][9 if selected and i < 3 else 10]]["fidelity"], level)
                        if active[i] else None for i, level in enumerate(LEVELS)]
        token9 = exact[prefix[number - 1][9]]["tokens"]
        token10 = exact[prefix[number - 1][10]]["tokens"]
        denominator = sum(active) * row["full_tokens"]
        baseline_cost = token10 * sum(active)
        policy_cost = baseline_cost - (3 * (token10 - token9) if selected else 0)
        table.append({"example_id": query, "switch": selected,
                      "logit_080_depth9": logits[number - 1].item(),
                      "baseline_success": baseline_flags, "policy_success": policy_flags,
                      "baseline_complete": all(value is not False for value in baseline_flags),
                      "policy_complete": all(value is not False for value in policy_flags),
                      "baseline_cost_fraction": baseline_cost / denominator,
                      "policy_cost_fraction": policy_cost / denominator})
        if number % 500 == 0:
            print(json.dumps({"evaluated": number, "total": len(rows)}), flush=True)
    result = {
        "queries": len(rows), "switches": sum(row["switch"] for row in table),
        "baseline_090": sum(bool(row["baseline_success"][3]) for row in table),
        "policy_090": sum(bool(row["policy_success"][3]) for row in table),
        "baseline_complete": sum(row["baseline_complete"] for row in table),
        "policy_complete": sum(row["policy_complete"] for row in table),
        "complete_gain": sum(not row["baseline_complete"] and row["policy_complete"] for row in table),
        "complete_break": sum(row["baseline_complete"] and not row["policy_complete"] for row in table),
        "anchor_baseline_success": [sum(row["baseline_success"][i] is True for row in table) for i in range(5)],
        "anchor_policy_success": [sum(row["policy_success"][i] is True for row in table) for i in range(5)],
        "baseline_mean_final_context_fraction": mean(row["baseline_cost_fraction"] for row in table),
        "policy_mean_final_context_fraction": mean(row["policy_cost_fraction"] for row in table),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_TRAIN_ONLY_GATE":
        raise ValueError("unfrozen protocol")
    rows = [row for row in read_jsonl(args.candidates) if fold(row["example_id"]) != 4]
    if len(rows) != 1421:
        raise ValueError("unexpected train-only population")
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    head = MaskValueHead(model_dim=512, layers=2, heads=8, dropout=.1)
    head.load_state_dict(torch.load(args.head, map_location="cpu", weights_only=True))
    head.to(device).eval()
    train = evaluate(rows, orders, cache, head, Path(args.exact_dir), device)
    passed = (train["baseline_090"] == train["policy_090"] and
              train["policy_complete"] >= train["baseline_complete"] and
              train["switches"] >= 30 and
              train["policy_mean_final_context_fraction"] < train["baseline_mean_final_context_fraction"])
    result = {"protocol": config["protocol"], "train_only_gate_passed": passed,
              "train": train, "fold4_611_read": False, "holdout_581_read": False,
              "internal300_read": False, "development_read": False,
              "confirmation_read": False, "new_target_calls": 0,
              "artifacts": {"config_sha256": sha256(args.config), "candidates_sha256": sha256(args.candidates),
                            "cache_sha256": sha256(args.cache), "head_sha256": sha256(args.head),
                            "rollouts_sha256": sha256(args.rollouts)}}
    write_metadata(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
