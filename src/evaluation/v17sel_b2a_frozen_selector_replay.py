from __future__ import annotations

import json
from pathlib import Path

import torch

from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.v17d1b_frozen_boundary_separability import stable_fold
from src.model.evidence_sufficiency_selector import EvidenceSufficiencySelector
from src.model.mask_value_head import MaskValueHead
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.training.train_v17h0_multianchor_cutoff import LEVELS, meets_fidelity, project


def exact_values(path: Path) -> tuple[list[float], list[int]]:
    fidelity = [0.0] * 4096
    tokens = [0] * 4096
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            mask = state_to_mask(row["state"])
            if mask in seen:
                raise ValueError(f"duplicate mask {path}")
            seen.add(mask)
            fidelity[mask] = float(row["fidelity"])
            tokens[mask] = int(row["tokens"])
    if len(seen) != 4096:
        raise ValueError(f"incomplete exact lattice {path}")
    return fidelity, tokens


def order_success(order: list[int], fidelity: list[float]) -> list[bool]:
    masks = [0]
    for packet in order:
        masks.append(masks[-1] | (1 << packet))
    return [any(meets_fidelity(fidelity[mask], level) for mask in masks) for level in LEVELS]


def choose(utilities: list[float], top_k: int, threshold: float) -> int:
    best = max(range(1, top_k + 1), key=lambda i: (utilities[i], -i))
    return best if utilities[best] - utilities[0] >= threshold else 0


def main() -> None:
    root = Path("results/v2_rank_then_cut")
    config_path = Path("configs/v17sel_b2a_frozen_selector_rank_clamp_replay.json")
    config = json.loads(config_path.read_text())
    if config["status"] != "FROZEN_DIAGNOSTIC_ONLY":
        raise ValueError("unfrozen protocol")
    candidates_path = root / "v14_train2863_top4_candidates.jsonl"
    cache_path = root / "v14_train2863_v13_embeddings.pt"
    v13_path = root / "v13_joint_retriever_seed20260918/checkpoints/step300/mask_retrieval_head.pt"
    v14_path = root / "v14_conservative_selector_seed20260918/selected_selector.pt"
    selected_path = root / "v14_conservative_selector_seed20260918/selected.json"
    candidates = list(read_jsonl(candidates_path))
    ids = [row["example_id"] for row in candidates]
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if cache["example_ids"] != ids:
        raise ValueError("candidate/embedding mismatch")
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(root / "v9_v8_train3163_beam8_rollouts.jsonl")}
    threshold = float(json.loads(selected_path.read_text())["threshold"])
    heldout = {i for i, query in enumerate(ids) if stable_fold(query, 5) == 0}
    if len(heldout) != 606:
        raise ValueError("unexpected H0 fold")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    retriever = MaskValueHead(model_dim=512, layers=2, heads=8, dropout=0.1)
    retriever.load_state_dict(torch.load(v13_path, map_location="cpu", weights_only=True))
    retriever.to(device).eval()
    selector = EvidenceSufficiencySelector()
    selector.load_state_dict(torch.load(v14_path, map_location="cpu", weights_only=True))
    selector.to(device).eval()
    exact_dir = root / "candidates5000_exact"
    rows = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for batch_start in range(0, len(candidates), 32):
            batch_stop = min(batch_start + 32, len(candidates))
            count = batch_stop - batch_start
            packets = cache["packets"][batch_start:batch_stop].to(device)
            questions = cache["questions"][batch_start:batch_stop].to(device)
            logits = torch.empty((count, 4096, 5), dtype=torch.float32)
            for mask_start in range(0, 4096, 512):
                masks = torch.arange(mask_start, mask_start + 512, device=device).repeat(count)
                example_indices = torch.arange(count, device=device).repeat_interleave(512)
                logits[:, mask_start:mask_start + 512] = retriever(packets, questions, masks, example_indices).reshape(count, 512, 5).float().cpu()
            for local in range(count):
                index = batch_start + local
                if index not in heldout:
                    continue
                query = ids[index]
                ranked = torch.argsort(logits[local, :, 3], descending=True, stable=True)[:20].tolist()
                original_top4 = [int(item["mask"]) for item in candidates[index]["candidates"][1:]]
                if ranked[:4] != original_top4:
                    raise ValueError(f"V14 top-four rank reproduction failed: {query}")
                fidelity, tokens = exact_values(exact_dir / f"{query}.jsonl")
                base = orders[query]
                masks = [0, *ranked]
                retrieval_logits = [logits[local, mask].tolist() for mask in masks]
                token_fractions = [tokens[mask] / max(1, tokens[-1]) for mask in masks]
                rank_buckets = [min(rank, 4) for rank in range(len(masks))]
                utility_logits = selector(
                    packets[local:local + 1], questions[local:local + 1],
                    torch.tensor(masks, device=device), torch.tensor(rank_buckets, device=device),
                    torch.tensor(retrieval_logits, device=device), torch.tensor(token_fractions, device=device),
                    torch.zeros(len(masks), device=device, dtype=torch.long),
                ).float()
                weights = torch.tensor([1, 1, 1, 2, 1], device=device)
                utilities = (torch.sigmoid(utility_logits) * weights).sum(1).cpu().tolist()
                success = [order_success(base if rank == 0 else project(base, mask), fidelity) for rank, mask in enumerate(masks)]
                if [item[3] for item in success[:5]] != [bool(item["success"][3]) for item in candidates[index]["candidates"]]:
                    raise ValueError(f"V14 oracle outcome reproduction failed: {query}")
                active = [bool(value) for value in candidates[index]["candidates"][0]["active"]]
                outcomes = {}
                for top_k in config["top_k"]:
                    chosen = choose(utilities, top_k, threshold)
                    oracle_pool = any(item[3] for item in success[:top_k + 1])
                    outcomes[str(top_k)] = {
                        "chosen_index": chosen,
                        "chosen_success": [value if is_active else None for value, is_active in zip(success[chosen], active)],
                        "complete": all(value or not is_active for value, is_active in zip(success[chosen], active)),
                        "oracle_pool_090": oracle_pool,
                    }
                rows.append({"example_id": query, "outcomes": outcomes, "chosen_utility": {str(k): utilities[outcomes[str(k)]["chosen_index"]] for k in config["top_k"]}})
            if batch_stop % 500 < 32 or batch_stop == len(candidates):
                print(json.dumps({"scored_population": batch_stop, "heldout_done": len(rows)}), flush=True)
    previous = {row["example_id"]: row for row in read_jsonl(root / "v17sel_b1_frozen_topk_coverage/per_query.jsonl")}
    if set(previous) != {row["example_id"] for row in rows}:
        raise ValueError("B1 population mismatch")
    metrics = {}
    for top_k in config["top_k"]:
        key = str(top_k)
        metrics[key] = {
            "oracle_pool_090": sum(row["outcomes"][key]["oracle_pool_090"] for row in rows),
            "selected_090": sum(bool(row["outcomes"][key]["chosen_success"][3]) for row in rows),
            "selected_complete": sum(row["outcomes"][key]["complete"] for row in rows),
            "switches_from_fallback": sum(row["outcomes"][key]["chosen_index"] != 0 for row in rows),
            "selected_beyond_original_top4": sum(row["outcomes"][key]["chosen_index"] > 4 for row in rows),
        }
        if metrics[key]["oracle_pool_090"] != sum(previous[row["example_id"]]["pool_090"][key] for row in rows):
            raise ValueError(f"SEL-B1 pool coverage mismatch at K={key}")
        metrics[key]["utilization"] = metrics[key]["selected_090"] / metrics[key]["oracle_pool_090"]
    for top_k in (10, 20):
        key = str(top_k)
        metrics[key]["repairs_vs_top4"] = sum(not row["outcomes"]["4"]["chosen_success"][3] and row["outcomes"][key]["chosen_success"][3] for row in rows)
        metrics[key]["breaks_vs_top4"] = sum(row["outcomes"]["4"]["chosen_success"][3] and not row["outcomes"][key]["chosen_success"][3] for row in rows)
    output = root / "v17sel_b2a_frozen_selector_replay"
    output.mkdir(exist_ok=True)
    write_jsonl(output / "per_query.jsonl", rows)
    summary = {
        "decision": "RANK_CLAMP_OUT_OF_SUPPORT_DIAGNOSTIC_ONLY",
        "queries": len(rows),
        "fixed_threshold": threshold,
        "by_top_k": metrics,
        "retrieval_scoring_masks_per_query": 4096,
        "selector_scored_candidates_by_k": {str(k): k + 1 for k in config["top_k"]},
        "development_used": False,
        "confirmation_used": False,
        "artifacts": {"config_sha256": sha256(config_path), "retriever_sha256": sha256(v13_path), "selector_sha256": sha256(v14_path), "candidates_sha256": sha256(candidates_path)},
    }
    write_metadata(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
