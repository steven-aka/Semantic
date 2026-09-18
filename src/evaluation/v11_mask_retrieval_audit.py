"""Audit direct 0.90-mask retrieval and V8 order projection on consumed development."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch

from src.data.schemas import read_jsonl, write_jsonl
from src.model.mask_value_head import MaskValueHead
from src.reproducibility import experiment_metadata, sha256, write_metadata


def project(order, mask):
    return tuple([packet for packet in order if mask & (1 << packet)] + [packet for packet in order if not mask & (1 << packet)])


def evaluate(order, row, oracle_tokens):
    values = row["mask_values"]
    tokens = [int(value["tokens"]) for value in values]
    attained = [int(value["attained_levels"]) for value in values]
    masks = [0]
    for packet in order:
        masks.append(masks[-1] | (1 << packet))
    active = len(row["attainable_levels"])
    selected = []
    for level in range(1, active + 1):
        feasible = [tokens[mask] for mask in masks if attained[mask] >= level]
        selected.append(min(feasible) if feasible else None)
    success = [value is not None for value in selected]
    cumulative = sum(value for value in selected if value is not None)
    regret = (
        (cumulative - sum(oracle_tokens)) / (active * tokens[-1]) if all(success) else None
    )
    return {"order": list(order), "success": success, "cumulative_tokens": cumulative, "regret": regret}


def summarize(details, field):
    rows = [row[field] for row in details]
    levels = [0.6, 0.7, 0.8, 0.9, 0.95]
    per_level = {}
    for index, level in enumerate(levels):
        active = [row for row in rows if len(row["success"]) > index]
        per_level[str(level)] = {"examples": len(active), "successes": sum(row["success"][index] for row in active)}
    regrets = [row["regret"] for row in rows if row["regret"] is not None]
    return {
        "per_level": per_level,
        "complete_trajectory_successes": sum(all(row["success"]) for row in rows),
        "complete_trajectory_success_fraction": sum(all(row["success"]) for row in rows) / len(rows),
        "feasible_examples": len(regrets),
        "mean_normalized_regret_feasible": mean(regrets) if regrets else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--v8-details", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mask-chunk-size", type=int, default=512)
    args = parser.parse_args()
    rows = list(read_jsonl(args.data))
    base = {row["example_id"]: row for row in read_jsonl(args.v8_details)}
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    if cache["example_ids"] != [row["example_id"] for row in rows]:
        raise ValueError("cache IDs differ from data")
    device = torch.device("cuda:0")
    head = MaskValueHead(model_dim=512, layers=2, heads=8, dropout=0.1)
    head.load_state_dict(torch.load(args.head, map_location="cpu", weights_only=True))
    head.to(device).eval()
    logits90 = torch.empty((len(rows), 4096), dtype=torch.float32)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, len(rows), args.batch_size):
            stop = min(start + args.batch_size, len(rows))
            packets = cache["packets"][start:stop].to(device)
            questions = cache["questions"][start:stop].to(device)
            count = stop - start
            for mask_start in range(0, 4096, args.mask_chunk_size):
                mask_stop = min(mask_start + args.mask_chunk_size, 4096)
                width = mask_stop - mask_start
                masks = torch.arange(mask_start, mask_stop, device=device).repeat(count)
                indices = torch.arange(count, device=device).repeat_interleave(width)
                output = head(packets, questions, masks, indices)[:, 3].reshape(count, width)
                logits90[start:stop, mask_start:mask_stop] = output.float().cpu()
    details = []
    for index, row in enumerate(rows):
        reference = base[row["example_id"]]
        base_order = tuple(reference["decoded_order"])
        oracle_tokens = [int(anchor["oracle_nested_tokens"]) for anchor in reference["anchors"]]
        ranked = torch.argsort(logits90[index], descending=True, stable=True).tolist()
        record = {"example_id": row["example_id"], "top_masks": ranked[:16]}
        baseline = evaluate(base_order, row, oracle_tokens)
        record["v8_baseline"] = baseline
        for size in (1, 4, 16):
            candidates = [baseline]
            candidates.extend(evaluate(project(base_order, mask), row, oracle_tokens) for mask in ranked[:size])
            best = min(candidates, key=lambda item: (-sum(item["success"]), item["cumulative_tokens"], item["order"]))
            if size == 1:
                record["top1_retrieval"] = candidates[1]
            record[f"top{size}_oracle"] = best
        truth = [int(value["attained_levels"]) >= 4 for value in row["mask_values"]]
        record["top1_mask_is_true_0.90"] = truth[ranked[0]]
        record["any_true_0.90_in_top4"] = any(truth[mask] for mask in ranked[:4])
        record["any_true_0.90_in_top16"] = any(truth[mask] for mask in ranked[:16])
        details.append(record)
    fields = ["v8_baseline", "top1_retrieval", "top1_oracle", "top4_oracle", "top16_oracle"]
    summary = {
        "complete": True,
        "scientific_role": "post-stop consumed-development retrieval audit; oracle top-k rows are nondeployable upper bounds",
        "results": {field: summarize(details, field) for field in fields},
        "retrieval": {
            "top1_true_0.90_masks": sum(row["top1_mask_is_true_0.90"] for row in details),
            "queries_with_true_0.90_in_top4": sum(row["any_true_0.90_in_top4"] for row in details),
            "queries_with_true_0.90_in_top16": sum(row["any_true_0.90_in_top16"] for row in details),
        },
        "artifacts": {"data_sha256": sha256(args.data), "cache_sha256": sha256(args.cache), "head_sha256": sha256(args.head), "v8_details_sha256": sha256(args.v8_details)},
    }
    write_jsonl(args.output, details)
    write_metadata(args.summary, summary)
    write_metadata(f"{args.output}.metadata.json", experiment_metadata(stage="v11_mask_retrieval_audit", **summary["artifacts"]))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
