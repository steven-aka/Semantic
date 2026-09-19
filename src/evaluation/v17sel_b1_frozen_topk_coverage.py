from __future__ import annotations

import json
from pathlib import Path

import torch

from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.v17d1b_frozen_boundary_separability import stable_fold
from src.model.mask_value_head import MaskValueHead
from src.reproducibility import sha256, write_metadata
from src.training.train_v17h0_multianchor_cutoff import meets_fidelity, project


def successful_order(order: list[int], success: list[bool]) -> bool:
    mask = 0
    if success[mask]:
        return True
    for packet in order:
        mask |= 1 << packet
        if success[mask]:
            return True
    return False


def exact_success(path: Path) -> list[bool]:
    success = [False] * 4096
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            mask = sum((1 << i) for i, value in enumerate(row["state"]) if value)
            if mask in seen:
                raise ValueError(f"duplicate exact mask {path}")
            seen.add(mask)
            success[mask] = meets_fidelity(float(row["fidelity"]), 0.90)
    if len(seen) != 4096:
        raise ValueError(f"incomplete exact lattice {path}")
    return success


def main() -> None:
    root = Path("results/v2_rank_then_cut")
    config_path = Path("configs/v17sel_b1_frozen_topk_coverage.json")
    config = json.loads(config_path.read_text())
    if config["status"] != "FROZEN_READ_ONLY":
        raise ValueError("unfrozen protocol")
    candidate_path = root / "v14_train2863_top4_candidates.jsonl"
    cache_path = root / "v14_train2863_v13_embeddings.pt"
    head_path = root / "v13_joint_retriever_seed20260918/checkpoints/step300/mask_retrieval_head.pt"
    candidates = list(read_jsonl(candidate_path))
    ids = [row["example_id"] for row in candidates]
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if ids != cache["example_ids"]:
        raise ValueError("embedding/candidate population mismatch")
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(root / "v9_v8_train3163_beam8_rollouts.jsonl")}
    indices = [i for i, query in enumerate(ids) if stable_fold(query, 5) == 0]
    if len(indices) != 606:
        raise ValueError("unexpected hash-heldout count")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    head = MaskValueHead(model_dim=512, layers=2, heads=8, dropout=0.1)
    head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
    head.to(device).eval()
    exact_dir = root / "candidates5000_exact"
    rows = []
    heldout = set(indices)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for batch_start in range(0, len(candidates), 32):
            batch_stop = min(batch_start + 32, len(candidates))
            count = batch_stop - batch_start
            packets = cache["packets"][batch_start:batch_stop].to(device)
            questions = cache["questions"][batch_start:batch_stop].to(device)
            scores = torch.empty((count, 4096), dtype=torch.float32)
            for mask_start in range(0, 4096, 512):
                masks = torch.arange(mask_start, mask_start + 512, device=device).repeat(count)
                example_indices = torch.arange(count, device=device).repeat_interleave(512)
                scores[:, mask_start:mask_start + 512] = head(packets, questions, masks, example_indices)[:, 3].reshape(count, 512).float().cpu()
            for local in range(count):
                index = batch_start + local
                if index not in heldout:
                    continue
                query = ids[index]
                ranked = torch.argsort(scores[local], descending=True, stable=True)[:20].tolist()
                original_top4 = [int(item["mask"]) for item in candidates[index]["candidates"][1:]]
                if ranked[:4] != original_top4:
                    raise ValueError(f"V14 top-four reproduction failed for {query}: {ranked[:4]} != {original_top4}")
                success = exact_success(exact_dir / f"{query}.jsonl")
                base = orders[query]
                base_success = successful_order(base, success)
                projection_success = [successful_order(project(base, mask), success) for mask in ranked]
                pool = {str(k): bool(base_success or any(projection_success[:k])) for k in config["top_k"]}
                if pool["4"] != any(item["success"][3] for item in candidates[index]["candidates"]):
                    raise ValueError(f"V14 oracle coverage mismatch for {query}")
                rows.append({
                    "example_id": query,
                    "v8_fallback": base_success,
                    "pool_090": pool,
                    "first_successful_rank": next((rank + 1 for rank, ok in enumerate(projection_success) if ok), None),
                    "global_exact_success": any(success),
                })
            if batch_stop % 500 < 32 or batch_stop == len(candidates):
                print(json.dumps({"scored_population": batch_stop, "total": len(candidates), "heldout_done": len(rows)}), flush=True)
    counts = {str(k): sum(row["pool_090"][str(k)] for row in rows) for k in config["top_k"]}
    if counts["4"] != sum(any(item["success"][3] for item in candidates[index]["candidates"]) for index in indices):
        raise ValueError("top-four count mismatch")
    output = root / "v17sel_b1_frozen_topk_coverage"
    output.mkdir(exist_ok=True)
    write_jsonl(output / "per_query.jsonl", rows)
    summary = {
        "decision": "FROZEN_PROPOSAL_POOL_DIAGNOSIS_ONLY",
        "queries": len(rows),
        "v8_fallback_090": sum(row["v8_fallback"] for row in rows),
        "oracle_pool_090_by_top_k": counts,
        "full_lattice_090": sum(row["global_exact_success"] for row in rows),
        "development_used": False,
        "confirmation_used": False,
        "artifact_sha256": {"config": sha256(config_path), "checkpoint": sha256(head_path), "embeddings": sha256(cache_path), "candidates": sha256(candidate_path)},
    }
    write_metadata(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
