"""Build clean-train top-10 candidates and exact oracle cost labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.schemas import read_jsonl, write_jsonl
from src.model.mask_value_head import MaskValueHead
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.training.train_v17h0_multianchor_cutoff import LEVELS, meets_fidelity, project


def outcome(order: list[int], fidelity: list[float], tokens: list[int], attainable: set[float]) -> dict:
    prefixes = [0]
    for packet in order:
        prefixes.append(prefixes[-1] | (1 << packet))
    if len(prefixes) != 13 or prefixes[-1] != 4095:
        raise ValueError("invalid projected order")
    success = [any(meets_fidelity(fidelity[mask], level) for mask in prefixes) for level in LEVELS]
    earliest = [min((tokens[mask] for mask in prefixes if meets_fidelity(fidelity[mask], level)), default=None) for level in LEVELS]
    # Ordered cutoffs must be monotone in packet count; fidelity itself may roll back.
    costs = {0: 0}
    for level in sorted(attainable):
        next_costs = {}
        for count, mask in enumerate(prefixes):
            if meets_fidelity(fidelity[mask], level):
                previous = [value for previous_count, value in costs.items() if previous_count <= count]
                if previous:
                    next_costs[count] = min(previous) + tokens[mask]
        costs = next_costs
    return {
        "success": success,
        "earliest_tokens": earliest,
        "complete": bool(costs),
        "oracle_ordered_complete_cumulative_tokens": min(costs.values()) if costs else None,
    }


def exact_values(path: Path) -> tuple[list[float], list[int]]:
    fidelity = [0.0] * 4096
    tokens = [0] * 4096
    seen = set()
    for row in read_jsonl(path):
        mask = state_to_mask(row["state"])
        if mask in seen:
            raise ValueError(f"duplicate exact mask {path}")
        seen.add(mask)
        fidelity[mask] = float(row["fidelity"])
        tokens[mask] = int(row["tokens"])
    if len(seen) != 4096:
        raise ValueError(f"incomplete exact lattice {path}")
    return fidelity, tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--role", choices=("train", "holdout"), default="train")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_APPROVED_TO_BUILD_TRAIN_ONLY":
        raise ValueError("C0 protocol is not frozen")
    expected = config["upstream"]
    source = expected["source_data"] if args.role == "train" else expected["holdout_source_data"]
    if (args.data, args.head, args.exact_dir) != (
        source, str(Path(expected["v13_checkpoint"]) / "mask_retrieval_head.pt"), expected["exact_dir"]
    ):
        raise ValueError("inputs differ from frozen C0 protocol")
    rows = list(read_jsonl(args.data))
    if args.limit:
        rows = rows[:args.limit]
    ids = [row["example_id"] for row in rows]
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    if cache["example_ids"][:len(rows)] != ids:
        raise ValueError("cache/data ID mismatch")
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    if not all(query in orders for query in ids):
        raise ValueError("missing V8 train-only rollout")
    device = torch.device("cuda:0")
    head = MaskValueHead(model_dim=512, layers=2, heads=8, dropout=0.1)
    head.load_state_dict(torch.load(args.head, map_location="cpu", weights_only=True))
    head.to(device).eval()
    output = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, len(rows), 16):
            stop = min(start + 16, len(rows))
            count = stop - start
            packets = cache["packets"][start:stop].to(device)
            questions = cache["questions"][start:stop].to(device)
            logits = torch.empty((count, 4096, 5), dtype=torch.float32)
            for mask_start in range(0, 4096, 256):
                masks = torch.arange(mask_start, mask_start + 256, device=device).repeat(count)
                indices = torch.arange(count, device=device).repeat_interleave(256)
                logits[:, mask_start:mask_start + 256] = head(packets, questions, masks, indices).reshape(count, 256, 5).float().cpu()
            for local, row in enumerate(rows[start:stop]):
                query = row["example_id"]
                base = [int(x) for x in orders[query]]
                ranked = torch.argsort(logits[local, :, 3], descending=True, stable=True)[:10].tolist()
                fidelity, tokens = exact_values(Path(args.exact_dir) / f"{query}.jsonl")
                attainable = {float(level) for level in row["attainable_levels"]}
                candidates = []
                for rank, mask in enumerate([0, *ranked]):
                    order = base if rank == 0 else project(base, mask)
                    candidates.append({
                        "rank": rank,
                        "mask": mask,
                        "retrieval_logits": logits[local, mask].tolist(),
                        "mask_tokens": tokens[mask],
                        "score_gap_090": float(logits[local, ranked[0], 3] - logits[local, mask, 3]) if rank else 0.0,
                        **outcome(order, fidelity, tokens, attainable),
                    })
                output.append({
                    "example_id": query,
                    "attainable_levels": sorted(attainable),
                    "full_tokens": tokens[-1],
                    "global_min_090_tokens": min((tokens[mask] for mask in range(4096) if meets_fidelity(fidelity[mask], .9)), default=None),
                    "unique_candidate_orders": len({tuple(base if item["rank"] == 0 else project(base, item["mask"])) for item in candidates}),
                    "candidates": candidates,
                })
            print(json.dumps({"built": len(output), "total": len(rows)}), flush=True)
    write_jsonl(args.output, output)
    write_metadata(args.manifest, {
        "complete": args.limit is None,
        "examples": len(output),
        "candidate_count": 11,
        "holdout_read": args.role == "holdout",
        "role": args.role,
        "config_sha256": sha256(args.config),
        "data_sha256": sha256(args.data),
        "cache_sha256": sha256(args.cache),
        "rollouts_sha256": sha256(args.rollouts),
        "head_sha256": sha256(args.head),
        "output_sha256": sha256(args.output),
    })


if __name__ == "__main__":
    main()
