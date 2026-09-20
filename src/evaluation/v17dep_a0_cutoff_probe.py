"""Train-only probe of unconditional 0.90 safe-window observability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.build_v17sel_c0_train_candidates import exact_values
from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.v17dep_a0_selector_probe import average_precision, train_linear
from src.reproducibility import sha256, write_metadata
from src.training.train_v17cut_b0_structured_fixed_v8 import prefix_masks
from src.training.train_v17h0_multianchor_cutoff import meets_fidelity


def score_decisions(scores: torch.Tensor, safe: torch.Tensor) -> dict:
    if scores.shape != safe.shape or scores.shape[1] != 13:
        raise ValueError("expected query by prefix scores and labels")
    reachable = safe.any(1)
    chosen = scores.argmax(1)
    selected_safe = safe[torch.arange(len(safe)), chosen]
    first = safe.int().argmax(1)
    hits = selected_safe & reachable
    early = (chosen < first) & reachable & ~hits
    after = (chosen > first) & reachable & ~hits
    widths = safe.sum(1)
    return {
        "queries": len(safe), "reachable_orders": int(reachable.sum()),
        "safe_prefix_prevalence": float(safe.float().mean()),
        "safe_prefix_average_precision": average_precision(list(zip(scores.flatten().tolist(), safe.flatten().tolist()))),
        "hit_safe_window": int(hits.sum()),
        "hit_fraction_given_reachable": float(hits.sum() / reachable.sum()) if bool(reachable.any()) else None,
        "early_miss": int(early.sum()), "post_first_rollback_miss": int(after.sum()),
        "one_safe_prefix_queries": int(((widths == 1) & reachable).sum()),
        "one_safe_prefix_hits": int((hits & (widths == 1)).sum()),
        "multiple_safe_prefix_queries": int(((widths > 1) & reachable).sum()),
        "multiple_safe_prefix_hits": int((hits & (widths > 1)).sum()),
        "chosen_depth_histogram": {str(depth): int((chosen == depth).sum()) for depth in range(13)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_TRAIN_ONLY_DIAGNOSTIC":
        raise ValueError("unfrozen protocol")
    source = [row for row in read_jsonl(args.candidates) if fold(row["example_id"]) != 4]
    if len(source) != 1421:
        raise ValueError("unexpected probe population")
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    cache_position = {query: index for index, query in enumerate(cache["example_ids"])}
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    safe_rows = []
    current_features = []
    history_features = []
    for position, row in enumerate(source, 1):
        query = row["example_id"]
        if query not in cache_position or query not in orders:
            raise ValueError("candidate/cache/rollout mismatch")
        masks = prefix_masks(orders[query])
        fidelity, tokens = exact_values(Path(args.exact_dir) / f"{query}.jsonl")
        flags = [meets_fidelity(fidelity[mask], .9) for mask in masks]
        if any(flags) != row["candidates"][0]["success"][3]:
            raise ValueError(f"0.90 exact-label mismatch {query}")
        safe_rows.append(flags)
        source_index = cache_position[query]
        packets = cache["packets"][source_index].float()
        question = cache["questions"][source_index].float()
        for depth, mask in enumerate(masks):
            bits = torch.tensor([(mask >> bit) & 1 for bit in range(12)], dtype=torch.float32)
            selected = (bits[:, None] * packets).sum(0) / max(1, depth)
            depth_and_cost = torch.tensor([depth / 12, tokens[mask] / row["full_tokens"]], dtype=torch.float32)
            current = torch.cat((question, selected, question * selected, depth_and_cost))
            last = packets[orders[query][depth - 1]] if depth else torch.zeros_like(question)
            current_features.append(current)
            history_features.append(torch.cat((current, last)))
        if position % 500 == 0:
            print(json.dumps({"loaded": position, "total": len(source)}), flush=True)
    safe = torch.tensor(safe_rows, dtype=torch.bool)
    folds = [fold(row["example_id"]) for row in source]
    variants = {"current": torch.stack(current_features), "current_plus_last_packet": torch.stack(history_features)}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["probe"]["seed"])
    predictions = {}
    summary = {}
    for name, x in variants.items():
        oof = torch.empty((len(source) * 13,), dtype=torch.float32)
        labels = safe.flatten()
        for heldout in range(4):
            train_query = [index for index, value in enumerate(folds) if value != heldout]
            test_query = [index for index, value in enumerate(folds) if value == heldout]
            train = torch.tensor([query_index * 13 + depth for query_index in train_query for depth in range(13)])
            test = torch.tensor([query_index * 13 + depth for query_index in test_query for depth in range(13)])
            mu, sigma, model = train_linear(x[train], labels[train], config["probe"], device)
            with torch.no_grad():
                oof[test] = model(((x[test] - mu) / sigma).to(device)).squeeze(-1).cpu()
        predictions[name] = oof.reshape(len(source), 13)
        summary[name] = score_decisions(predictions[name], safe)
    # Empirical depth prior is fitted on the other three folds for each query.
    prior = torch.empty((len(source), 13), dtype=torch.float32)
    for heldout in range(4):
        train = torch.tensor([index for index, value in enumerate(folds) if value != heldout])
        test = torch.tensor([index for index, value in enumerate(folds) if value == heldout])
        rate = safe[train].float().mean(0)
        prior[test] = rate[None, :]
    summary["depth_only"] = score_decisions(prior, safe)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "cutoff_oof_queries.jsonl", [
        {"example_id": row["example_id"], "safe_090": safe[index].tolist(),
         "depth_prior": prior[index].tolist(),
         "current_score": predictions["current"][index].tolist(),
         "current_plus_last_packet_score": predictions["current_plus_last_packet"][index].tolist()}
        for index, row in enumerate(source)])
    result = {"protocol": config["protocol"], "role": "train-only 0.90 upper-bound diagnostic, not a multi-anchor deployed cutoff",
              "query_grouped_oof": summary,
              "fold4_611_read": False, "holdout_581_read": False,
              "internal300_read": False, "development_read": False,
              "confirmation_read": False, "new_target_calls": 0,
              "artifacts": {"config_sha256": sha256(args.config), "candidates_sha256": sha256(args.candidates),
                            "cache_sha256": sha256(args.cache), "rollouts_sha256": sha256(args.rollouts)}}
    write_metadata(out / "cutoff_summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
