"""Read-only replay without exact-derived attainability at cutoff inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.build_v17sel_c0_train_candidates import exact_values
from src.data.schemas import read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata
from src.training.train_v17cut_b0_structured_fixed_v8 import StructuredCutoff, best_stop_vector, prefix_masks
from src.training.train_v17h0_multianchor_cutoff import LEVELS, meets_fidelity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--original-decisions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = [row for row in read_jsonl(args.candidates) if fold(row["example_id"]) == 4]
    if len(rows) != 611:
        raise ValueError("expected design-exposed 611")
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    index = {query: position for position, query in enumerate(cache["example_ids"])}
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    prior = {row["example_id"]: row for row in read_jsonl(args.original_decisions)}
    if set(prior) != {row["example_id"] for row in rows}:
        raise ValueError("original decisions mismatch")
    model = StructuredCutoff().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    details = []
    for row in rows:
        query = row["example_id"]
        position = index[query]
        masks = prefix_masks(orders[query])
        fidelity, tokens = exact_values(Path(args.exact_dir) / f"{query}.jsonl")
        prefix_tokens = torch.tensor([[tokens[mask] for mask in masks]], dtype=torch.float32)
        with torch.no_grad():
            scores = model(cache["questions"][position:position + 1],
                           cache["packets"][position:position + 1],
                           prefix_tokens, torch.tensor([row["full_tokens"]], dtype=torch.float32),
                           torch.tensor([masks], dtype=torch.int32))[0]
        all_five = best_stop_vector(scores)
        active = [i for i, level in enumerate(LEVELS) if level in row["attainable_levels"]]
        old = tuple(prior[query]["stop_vector"])
        if len(old) != len(active):
            raise ValueError("prior active stop vector mismatch")
        new = tuple(all_five[i] for i in active)
        old_success = [meets_fidelity(fidelity[masks[depth]], LEVELS[anchor])
                       for anchor, depth in zip(active, old)]
        new_success = [meets_fidelity(fidelity[masks[depth]], LEVELS[anchor])
                       for anchor, depth in zip(active, new)]
        if bool(all(old_success)) != prior[query]["complete"]:
            raise ValueError("original replay does not reproduce Complete")
        details.append({"example_id": query, "active_level_count": len(active),
                        "oracle_mask_path": old, "all_five_path": all_five,
                        "all_five_path_on_active_levels": new,
                        "oracle_mask_success": old_success, "all_five_success_on_active_levels": new_success})
    def counts(key: str) -> dict:
        return {"success_090": sum(item[key][3] for item in details),
                "complete_on_attainable_levels": sum(all(item[key]) for item in details)}
    summary = {
        "role": "read-only diagnostic; all-five requested-level inference versus exact-derived attainable-level inference",
        "queries": len(details), "four_level_queries": sum(item["active_level_count"] == 4 for item in details),
        "oracle_attainability_mask": counts("oracle_mask_success"),
        "all_five_decode_scored_on_attainable_levels": counts("all_five_success_on_active_levels"),
        "changed_active_stop_vectors": sum(item["oracle_mask_path"] != item["all_five_path_on_active_levels"] for item in details),
        "changed_four_level_stop_vectors": sum(item["active_level_count"] == 4 and
                                               item["oracle_mask_path"] != item["all_five_path_on_active_levels"]
                                               for item in details),
        "candidate_policy_changed": False, "model_retrained": False,
        "holdout_581_read": False, "internal300_read": False, "development_read": False,
        "confirmation_read": False, "new_target_calls": 0,
        "artifacts": {"checkpoint_sha256": sha256(args.checkpoint), "candidates_sha256": sha256(args.candidates),
                      "cache_sha256": sha256(args.cache), "original_decisions_sha256": sha256(args.original_decisions)},
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "per_query.jsonl", details)
    write_metadata(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
