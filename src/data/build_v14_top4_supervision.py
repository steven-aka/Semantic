from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.model.mask_value_head import MaskValueHead
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask


LEVELS = (0.60, 0.70, 0.80, 0.90, 0.95)


def project(order: list[int], mask: int) -> list[int]:
    return [packet for packet in order if mask & (1 << packet)] + [
        packet for packet in order if not mask & (1 << packet)
    ]


def trajectory_outcome(order, fidelity, tokens, active_levels):
    prefixes = [0]
    for packet in order:
        prefixes.append(prefixes[-1] | (1 << packet))
    success = []
    selected_tokens = []
    for level in LEVELS:
        feasible = [tokens[mask] for mask in prefixes if fidelity[mask] + 1e-12 >= level]
        success.append(bool(feasible))
        selected_tokens.append(min(feasible) if feasible else None)
    active = [level in active_levels for level in LEVELS]
    complete = all(ok for ok, is_active in zip(success, active) if is_active)
    cumulative = sum(value for value, is_active in zip(selected_tokens, active) if is_active and value is not None)
    return {
        "success": success,
        "active": active,
        "complete": complete,
        "cumulative_tokens": cumulative,
        "selected_tokens": selected_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen V13 top-4 trajectory supervision")
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    rows = list(read_jsonl(args.data))
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    ids = [row["example_id"] for row in rows]
    if cache["example_ids"] != ids or any(example_id not in orders for example_id in ids):
        raise ValueError("data, cache, and rollout IDs do not align")
    device = torch.device("cuda:0")
    head = MaskValueHead(model_dim=512, layers=2, heads=8, dropout=0.1)
    head.load_state_dict(torch.load(args.head, map_location="cpu", weights_only=True))
    head.to(device).eval()
    output = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, len(rows), args.batch_size):
            stop = min(start + args.batch_size, len(rows))
            packets = cache["packets"][start:stop].to(device)
            questions = cache["questions"][start:stop].to(device)
            count = stop - start
            scores = torch.empty((count, 4096), dtype=torch.float32)
            all_logits = torch.empty((count, 4096, 5), dtype=torch.float32)
            for mask_start in range(0, 4096, 512):
                width = min(512, 4096 - mask_start)
                masks = torch.arange(mask_start, mask_start + width, device=device).repeat(count)
                indices = torch.arange(count, device=device).repeat_interleave(width)
                logits = head(packets, questions, masks, indices).reshape(count, width, 5).float().cpu()
                all_logits[:, mask_start : mask_start + width] = logits
                scores[:, mask_start : mask_start + width] = logits[:, :, 3]
            for local, row in enumerate(rows[start:stop]):
                ranked = torch.argsort(scores[local], descending=True, stable=True)[:4].tolist()
                masks = [0, *ranked]
                exact = list(read_jsonl(Path(args.exact_dir) / f"{row['example_id']}.jsonl", ExactSearchResult))
                by_mask = {state_to_mask(item.state): item for item in exact}
                fidelity = [float(by_mask[mask].fidelity) for mask in range(4096)]
                tokens = [int(by_mask[mask].tokens) for mask in range(4096)]
                base_order = orders[row["example_id"]]
                candidates = []
                for rank, mask in enumerate(masks):
                    order = base_order if rank == 0 else project(base_order, mask)
                    outcome = trajectory_outcome(order, fidelity, tokens, set(row["attainable_levels"]))
                    candidates.append({
                        "rank": rank,
                        "mask": mask,
                        "retrieval_logits": all_logits[local, mask].tolist(),
                        "mask_tokens": tokens[mask],
                        **outcome,
                    })
                output.append({
                    "example_id": row["example_id"],
                    "full_tokens": tokens[-1],
                    "candidates": candidates,
                })
            print(json.dumps({"built": stop, "total": len(rows)}), flush=True)
    write_jsonl(args.output, output)
    write_metadata(args.manifest, {
        "complete": True,
        "examples": len(output),
        "candidate_count": 5,
        "candidate_definition": "V8 fallback plus frozen V13 top-4 mask projections",
        "data_sha256": sha256(args.data),
        "cache_sha256": sha256(args.cache),
        "rollouts_sha256": sha256(args.rollouts),
        "head_sha256": sha256(args.head),
        "output_sha256": sha256(args.output),
        "locked_roles_used": False,
    })


if __name__ == "__main__":
    main()
