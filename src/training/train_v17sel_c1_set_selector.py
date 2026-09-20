"""Train frozen-upstream top-4 and top-10 set selectors on clean-train queries."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl, write_jsonl
from src.model.topk_set_selector import TopKSetSelector, successful_set_loss
from src.reproducibility import sha256, write_metadata


def load_features(candidates_path: str, cache_path: str) -> tuple[list[dict], torch.Tensor, torch.Tensor]:
    rows = list(read_jsonl(candidates_path))
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if cache["example_ids"] != [row["example_id"] for row in rows]:
        raise ValueError("candidate/cache population mismatch")
    masks = torch.tensor([[item["mask"] for item in row["candidates"]] for row in rows], dtype=torch.long)
    if masks.shape[1] != 11:
        raise ValueError("expected fallback plus top10")
    bits = ((masks[:, :, None] >> torch.arange(12)) & 1).float()
    packets = cache["packets"].float()
    question = cache["questions"].float()
    selected = torch.einsum("bcn,bnd->bcd", bits, packets) / bits.sum(2, keepdim=True).clamp_min(1)
    remaining = torch.einsum("bcn,bnd->bcd", 1 - bits, packets) / (12 - bits.sum(2, keepdim=True)).clamp_min(1)
    q = question[:, None, :].expand(-1, 11, -1)
    logits = torch.tensor([[item["retrieval_logits"] for item in row["candidates"]] for row in rows], dtype=torch.float32)
    rank = torch.arange(11).float()[None, :, None].expand(len(rows), -1, -1) / 10
    gap = torch.tensor([[item["score_gap_090"] for item in row["candidates"]] for row in rows], dtype=torch.float32)[:, :, None]
    tokens = torch.tensor([[item["mask_tokens"] / row["full_tokens"] for item in row["candidates"]] for row in rows], dtype=torch.float32)[:, :, None]
    count = bits.sum(2, keepdim=True) / 12
    features = torch.cat((q, selected, remaining, q * selected, logits, rank, gap, tokens, count), dim=2)
    if features.shape[2] != 4 * 512 + 9 or not torch.isfinite(features).all():
        raise ValueError("invalid selector features")
    unique = torch.tensor([[bool(item["unique_order"]) for item in row["candidates"]] for row in rows], dtype=torch.bool)
    if not unique[:, 0].all() or unique.sum(1).min() < 1:
        raise ValueError("invalid trajectory deduplication")
    return rows, features, unique


def metrics(rows: list[dict], selected: list[int], k: int) -> dict:
    if len(rows) != len(selected):
        raise ValueError("selection length mismatch")
    success = [bool(row["candidates"][index]["success"][3]) for row, index in zip(rows, selected)]
    complete = [bool(row["candidates"][index]["complete"]) for row, index in zip(rows, selected)]
    penalized = [
        row["candidates"][index]["earliest_tokens"][3] / row["full_tokens"] if good else 1.0
        for row, index, good in zip(rows, selected, success)
    ]
    successful_tokens = [row["candidates"][index]["earliest_tokens"][3] for row, index, good in zip(rows, selected, success) if good]
    return {
        "queries": len(rows), "pool_k": k, "success_090": sum(success), "complete": sum(complete),
        "mean_penalized_090_token_fraction": sum(penalized) / len(rows),
        "mean_earliest_090_tokens_successful": sum(successful_tokens) / len(successful_tokens) if successful_tokens else None,
        "selected_rank_histogram": {str(rank): selected.count(rank) for rank in range(k + 1)},
    }


def evaluate(model: TopKSetSelector, features: torch.Tensor, unique: torch.Tensor, indices: list[int], rows: list[dict], k: int, device: torch.device) -> tuple[dict, list[int]]:
    model.eval()
    choices = []
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            subset = indices[start:start + 128]
            scores = model(features[subset, :k + 1].to(device), unique[subset, :k + 1].to(device))
            choices.extend(torch.argmax(scores, dim=1).cpu().tolist())
    return metrics([rows[i] for i in indices], choices, k), choices


def run_arm(k: int, config: dict, rows: list[dict], features: torch.Tensor, unique: torch.Tensor, output: Path, device: torch.device) -> tuple[dict, list[int]]:
    seed = int(config["optimization"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    train_ids = [i for i, row in enumerate(rows) if fold(row["example_id"]) != 4]
    valid_ids = [i for i, row in enumerate(rows) if fold(row["example_id"]) == 4]
    train_success = torch.tensor([[item["success"][3] and item["unique_order"] for item in row["candidates"][:k + 1]] for row in rows], dtype=torch.bool)
    model = TopKSetSelector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["optimization"]["learning_rate"], weight_decay=config["optimization"]["weight_decay"])
    steps = int(config["optimization"]["steps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_ids), generator=generator).tolist()
    cursor = 0
    history = []
    checkpoints = []
    output.mkdir(parents=True, exist_ok=True)
    for step in range(1, steps + 1):
        model.train()
        batch = []
        while len(batch) < int(config["optimization"]["batch_size"]):
            if cursor == len(order):
                order = torch.randperm(len(train_ids), generator=generator).tolist()
                cursor = 0
            take = min(int(config["optimization"]["batch_size"]) - len(batch), len(order) - cursor)
            batch.extend(train_ids[index] for index in order[cursor:cursor + take])
            cursor += take
        selected = torch.tensor(batch)
        scores = model(features[selected, :k + 1].to(device), unique[selected, :k + 1].to(device))
        loss = successful_set_loss(scores, train_success[selected].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step in config["optimization"]["validation_checkpoints"]:
            validation, choices = evaluate(model, features, unique, valid_ids, rows, k, device)
            checkpoint = output / f"step{step}.pt"
            torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, checkpoint)
            record = {"step": step, "train_loss": float(loss.detach()), "validation": validation, "checkpoint": str(checkpoint)}
            checkpoints.append(record)
            history.append(record)
            write_jsonl(output / "history.jsonl", history)
            print(json.dumps({"arm": k, **record}), flush=True)
    best = max(checkpoints, key=lambda record: (
        record["validation"]["success_090"], record["validation"]["complete"],
        -record["validation"]["mean_penalized_090_token_fraction"], -record["step"],
    ))
    model.load_state_dict(torch.load(best["checkpoint"], map_location=device, weights_only=True))
    validation, choices = evaluate(model, features, unique, valid_ids, rows, k, device)
    write_metadata(output / "selected.json", best)
    return validation, choices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_APPROVED_TO_BUILD_TRAIN_ONLY":
        raise ValueError("C0 protocol is not frozen")
    rows, features, unique = load_features(args.candidates, args.cache)
    if len(rows) != 2032 or len({row["example_id"] for row in rows}) != 2032:
        raise ValueError("unexpected clean-train population")
    valid_ids = [i for i, row in enumerate(rows) if fold(row["example_id"]) == 4]
    if not valid_ids:
        raise ValueError("empty selector validation split")
    device = torch.device("cuda:0")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    decisions = {}
    for arm, k in config["candidate_budget"]["arms"].items():
        result, selected = run_arm(k, config, rows, features, unique, output / arm, device)
        results[arm] = result
        decisions[arm] = selected
    validation_rows = [rows[i] for i in valid_ids]
    fallback = metrics(validation_rows, [0] * len(valid_ids), 0)
    rank1 = metrics(validation_rows, [1] * len(valid_ids), 1)
    top4 = results["top4"]
    top10 = results["top10"]
    repairs = sum(not row["candidates"][a]["success"][3] and row["candidates"][b]["success"][3] for row, a, b in zip(validation_rows, decisions["top4"], decisions["top10"]))
    breaks = sum(row["candidates"][a]["success"][3] and not row["candidates"][b]["success"][3] for row, a, b in zip(validation_rows, decisions["top4"], decisions["top10"]))
    gate = config["train_only_gate"]
    passed = (
        top10["success_090"] - top4["success_090"] >= gate["top10_vs_top4_same_architecture_min_net_successes"]
        and top10["success_090"] - fallback["success_090"] >= gate["top10_vs_fixed_v8_fallback_min_net_successes"]
        and top10["complete"] - top4["complete"] >= gate["top10_vs_top4_complete_min_delta"]
        and top10["mean_penalized_090_token_fraction"] - top4["mean_penalized_090_token_fraction"] <= gate["top10_vs_top4_mean_penalized_090_token_fraction_max_delta"]
    )
    summary = {
        "decision": "GO_SEL_C1_ONE_DESIGN_EXPOSED_HOLDOUT_READ" if passed else "STOP_SEL_C1_TRAIN_ONLY_GATE",
        "selector_validation_queries": len(valid_ids), "top4": top4, "top10": top10,
        "v8_fallback": fallback, "v13_rank1": rank1, "repairs": repairs, "breaks": breaks,
        "holdout_read": False, "development_read": False, "confirmation_read": False,
        "artifacts": {"config_sha256": sha256(args.config), "candidates_sha256": sha256(args.candidates), "cache_sha256": sha256(args.cache)},
    }
    write_metadata(output / "summary.json", summary)
    write_jsonl(output / "validation_choices.jsonl", [
        {"example_id": row["example_id"], "top4_selected_rank": a, "top10_selected_rank": b}
        for row, a, b in zip(validation_rows, decisions["top4"], decisions["top10"])
    ])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
