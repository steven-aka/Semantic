"""Ask whether the existing V13 mask-value head supports safe online stopping."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.build_v17sel_c0_train_candidates import exact_values
from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.v17dep_a0_selector_probe import average_precision, train_linear
from src.model.mask_value_head import MaskValueHead
from src.reproducibility import sha256, write_metadata
from src.training.train_v17cut_b0_structured_fixed_v8 import prefix_masks
from src.training.train_v17h0_multianchor_cutoff import meets_fidelity


def decision_metrics(scores: torch.Tensor, safe: torch.Tensor, token_fraction: torch.Tensor,
                     *, first_crossing: bool) -> dict:
    if first_crossing:
        crossing = scores >= 0
        chosen = crossing.int().argmax(1)
        chosen = torch.where(crossing.any(1), chosen, torch.full_like(chosen, 12))
    else:
        chosen = scores.argmax(1)
    row = torch.arange(len(safe))
    reachable = safe.any(1)
    one_safe = safe.sum(1) == 1
    return {
        "queries": len(safe),
        "reachable": int(reachable.sum()),
        "success_090": int(safe[row, chosen].sum()),
        "hit_given_reachable": int((safe[row, chosen] & reachable).sum()),
        "one_safe_prefix_hits": int((safe[row, chosen] & one_safe).sum()),
        "one_safe_prefix_queries": int(one_safe.sum()),
        "mean_context_fraction": float(token_fraction[row, chosen].mean()),
        "mean_context_fraction_on_success": float(token_fraction[row, chosen][safe[row, chosen]].mean()),
        "mean_failure_penalized_context_fraction": float(torch.where(safe[row, chosen], token_fraction[row, chosen], 1.0).mean()),
        "chosen_depth_histogram": {str(depth): int((chosen == depth).sum()) for depth in range(13)},
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
    if config["status"] != "FROZEN_TRAIN_ONLY_DIAGNOSTIC":
        raise ValueError("unfrozen signal audit")
    rows = [row for row in read_jsonl(args.candidates) if fold(row["example_id"]) != 4]
    if len(rows) != 1421:
        raise ValueError("unexpected train-only count")
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    positions = {query: index for index, query in enumerate(cache["example_ids"])}
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    masks, labels, token_fractions = [], [], []
    for index, row in enumerate(rows, 1):
        query = row["example_id"]
        if query not in positions or query not in orders:
            raise ValueError("candidate/cache/rollout mismatch")
        prefix = prefix_masks(orders[query])
        fidelity, tokens = exact_values(Path(args.exact_dir) / f"{query}.jsonl")
        flags = [meets_fidelity(fidelity[mask], .9) for mask in prefix]
        if any(flags) != row["candidates"][0]["success"][3]:
            raise ValueError(f"cached V8 0.90 mismatch: {query}")
        masks.append(prefix)
        labels.append(flags)
        token_fractions.append([tokens[mask] / row["full_tokens"] for mask in prefix])
        if index % 500 == 0:
            print(json.dumps({"loaded": index, "total": len(rows)}), flush=True)
    masks_tensor = torch.tensor(masks, dtype=torch.long)
    safe = torch.tensor(labels, dtype=torch.bool)
    token_fraction = torch.tensor(token_fractions, dtype=torch.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MaskValueHead(model_dim=512, layers=2, heads=8, dropout=.1)
    model.load_state_dict(torch.load(args.head, map_location="cpu", weights_only=True))
    model.to(device).eval()
    head_logits = torch.empty((len(rows), 13, 5), dtype=torch.float32)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for start in range(0, len(rows), 32):
            subset = rows[start:start + 32]
            indices = torch.tensor([positions[row["example_id"]] for row in subset])
            packets = cache["packets"][indices].float().to(device)
            questions = cache["questions"][indices].float().to(device)
            mask_batch = masks_tensor[start:start + len(subset)].to(device)
            example_indices = torch.arange(len(subset), device=device).repeat_interleave(13)
            output = model(packets, questions, mask_batch.flatten(), example_indices)
            head_logits[start:start + len(subset)] = output.reshape(len(subset), 13, 5).float().cpu()
    depth = torch.arange(13, dtype=torch.float32)[None, :, None].expand(len(rows), -1, -1) / 12
    delta = torch.cat((torch.zeros((len(rows), 1, 1)),
                       (head_logits[:, 1:, 3] - head_logits[:, :-1, 3])[..., None]), dim=1)
    feature = torch.cat((depth, token_fraction[..., None], head_logits, delta), dim=2).flatten(0, 1)
    folds = [fold(row["example_id"]) for row in rows]
    fitted = torch.empty((len(rows) * 13,), dtype=torch.float32)
    torch.manual_seed(config.get("seed", 20260920))
    for heldout in range(4):
        train = torch.tensor([query * 13 + depth for query, value in enumerate(folds) if value != heldout for depth in range(13)])
        test = torch.tensor([query * 13 + depth for query, value in enumerate(folds) if value == heldout for depth in range(13)])
        mu, sigma, probe = train_linear(feature[train], safe.flatten()[train],
                                         {"epochs": 100, "batch_size": 512, "learning_rate": .01,
                                          "weight_decay": .001, "seed": 20260920}, device)
        with torch.no_grad():
            fitted[test] = probe(((feature[test] - mu) / sigma).to(device)).squeeze(-1).cpu()
    fitted = fitted.reshape(len(rows), 13)
    direct = head_logits[:, :, 3]
    fixed_depth10 = torch.full_like(direct, -1.0)
    fixed_depth10[:, 10] = 1.0
    result = {
        "protocol": config["protocol"], "role": "train-only OOF signal audit; V13 has query ancestry exposure",
        "positive_prefix_prevalence": float(safe.float().mean()),
        "direct_head_average_precision": average_precision(list(zip(direct.flatten().tolist(), safe.flatten().tolist()))),
        "fitted_average_precision": average_precision(list(zip(fitted.flatten().tolist(), safe.flatten().tolist()))),
        "online": {
            "fixed_depth10": decision_metrics(fixed_depth10, safe, token_fraction, first_crossing=True),
            "direct_v13_first_crossing": decision_metrics(direct, safe, token_fraction, first_crossing=True),
            "fitted_oof_first_crossing": decision_metrics(fitted, safe, token_fraction, first_crossing=True),
        },
        "offline_argmax_upper_bound": {
            "direct_v13": decision_metrics(direct, safe, token_fraction, first_crossing=False),
            "fitted_oof": decision_metrics(fitted, safe, token_fraction, first_crossing=False),
        },
        "fold4_611_read": False, "holdout_581_read": False, "internal300_read": False,
        "development_read": False, "confirmation_read": False, "new_target_calls": 0,
        "artifacts": {"config_sha256": sha256(args.config), "candidates_sha256": sha256(args.candidates),
                      "cache_sha256": sha256(args.cache), "head_sha256": sha256(args.head),
                      "rollouts_sha256": sha256(args.rollouts)},
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train_oof_queries.jsonl", [
        {"example_id": row["example_id"], "safe_090": safe[index].tolist(),
         "token_fraction": token_fraction[index].tolist(),
         "direct_090_logit": direct[index].tolist(), "fitted_oof_logit": fitted[index].tolist()}
        for index, row in enumerate(rows)])
    write_metadata(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
