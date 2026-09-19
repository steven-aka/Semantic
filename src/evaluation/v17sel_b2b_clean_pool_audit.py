"""One-shot lineage-clean oracle pool coverage, after every upstream endpoint is frozen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.v17sel_b1_frozen_topk_coverage import exact_success, successful_order
from src.model.mask_value_head import MaskValueHead
from src.reproducibility import sha256, write_metadata
from src.training.train_v17h0_multianchor_cutoff import project


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_APPROVED_TO_BUILD_AND_RUN":
        raise ValueError("pool protocol is not frozen")
    rows = list(read_jsonl(args.data))
    ids = [row["example_id"] for row in rows]
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    if cache["example_ids"] != ids or len(ids) != len(set(ids)):
        raise ValueError("holdout cache population mismatch")
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    if set(orders) != set(ids):
        raise ValueError("V8 holdout rollout population mismatch")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    head = MaskValueHead(model_dim=512, layers=2, heads=8, dropout=0.1)
    head.load_state_dict(torch.load(args.head, map_location="cpu", weights_only=True))
    head.to(device).eval()
    output = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for start in range(0, len(ids), 16):
            stop = min(start + 16, len(ids))
            count = stop - start
            packets = cache["packets"][start:stop].to(device)
            questions = cache["questions"][start:stop].to(device)
            scores = torch.empty((count, 4096), dtype=torch.float32)
            for mask_start in range(0, 4096, 256):
                masks = torch.arange(mask_start, mask_start + 256, device=device).repeat(count)
                indices = torch.arange(count, device=device).repeat_interleave(256)
                scores[:, mask_start:mask_start + 256] = head(packets, questions, masks, indices)[:, 3].reshape(count, 256).float().cpu()
            for local, query in enumerate(ids[start:stop]):
                ranked = torch.argsort(scores[local], descending=True, stable=True)[:20].tolist()
                success = exact_success(Path(args.exact_dir) / f"{query}.jsonl")
                base = orders[query]
                if sorted(base) != list(range(12)):
                    raise ValueError(f"invalid V8 order: {query}")
                base_success = successful_order(base, success)
                projections = [successful_order(project(base, mask), success) for mask in ranked]
                output.append({
                    "example_id": query,
                    "v8_fallback": base_success,
                    "pool_090": {str(k): bool(base_success or any(projections[:k])) for k in config["top_k"]},
                    "first_successful_projection_rank": next((i + 1 for i, ok in enumerate(projections) if ok), None),
                    "global_exact_success": any(success),
                })
            print(json.dumps({"scored": len(output), "total": len(ids)}), flush=True)
    if len(output) != 581:
        raise ValueError(f"unexpected holdout size: {len(output)}")
    counts = {str(k): sum(row["pool_090"][str(k)] for row in output) for k in config["top_k"]}
    if not counts["4"] <= counts["10"] <= counts["20"]:
        raise ValueError("pool coverage must be monotone")
    n = len(output)
    gain = (counts["10"] - counts["4"]) / n
    absolute = counts["10"] / n
    primary = config["primary_gate"]
    stop = config["clear_stop"]
    decision = (
        "GO_SEL_C0_TOPK_SET_SELECTOR_PROTOCOL" if absolute >= primary["top10_absolute_coverage_min"] and gain >= primary["top10_minus_top4_min"]
        else "STOP_TOPK_POOL_BRANCH" if absolute < stop["top10_absolute_coverage_below"] or gain < stop["top10_minus_top4_below"]
        else "INCONCLUSIVE_NO_SELECTOR_TRAINING"
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "per_query.jsonl", output)
    write_metadata(out / "summary.json", {
        "decision": decision,
        "queries": n,
        "oracle_pool_090_by_top_k": counts,
        "v8_fallback_090": sum(row["v8_fallback"] for row in output),
        "global_exact_success": sum(row["global_exact_success"] for row in output),
        "top10_minus_top4_fraction": gain,
        "top10_absolute_fraction": absolute,
        "development_used": False,
        "confirmation_used": False,
        "selector_trained": False,
        "cutoff_trained": False,
        "artifact_sha256": {name: sha256(path) for name, path in {
            "config": args.config, "data": args.data, "cache": args.cache,
            "head": args.head, "rollouts": args.rollouts,
        }.items()},
    })
    print(json.dumps({"decision": decision, "counts": counts, "queries": n}, indent=2), flush=True)


if __name__ == "__main__":
    main()
