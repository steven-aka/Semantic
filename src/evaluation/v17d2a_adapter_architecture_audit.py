from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.schemas import ExactSearchResult, read_jsonl
from src.evaluation.v16c0_first_irreversible_divergence import trajectory_levels
from src.evaluation.v17d1a_scoring_counterfactual_audit import metric_for_order
from src.model.v17d2_branch_adapter import Branch090AdaptedViabilityHead
from src.reproducibility import sha256, write_metadata
from src.training.train_v17b1_multi_anchor_viability import load_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen V17-D2A branch adapter protocol")
    parser.add_argument("--protocol-config", required=True)
    parser.add_argument("--internal-data", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--viability-head", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol_config).read_text())
    if protocol["status"] != "FROZEN_PENDING_ARCHITECTURE_AUDIT":
        raise ValueError("protocol is not in the audit state")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_policy(args.checkpoint, device, protocol["architecture"]["viability_hidden_dim"])
    model.viability_head.load_state_dict(torch.load(args.viability_head, map_location=device, weights_only=True))
    feature_dim = model.viability_head.viability[0].in_features
    original_head = model.viability_head
    wrapped = Branch090AdaptedViabilityHead(
        original_head, feature_dim, protocol["architecture"]["rank"], protocol["architecture"]["layer_norm_eps"]
    ).to(device)
    model.viability_head = wrapped
    model.eval()

    generator = torch.Generator(device=device).manual_seed(protocol["training_if_authorized"]["seed"])
    features = torch.randn(7, 12, feature_dim, generator=generator, device=device)
    active = torch.randint(0, 2, (7, 12, 5), generator=generator, device=device, dtype=torch.bool)
    active[..., 3] = True
    with torch.no_grad():
        original_logits, original_residual = original_head(features, active)
        routed_logits, routed_residual = wrapped(features, active)
    logit_diff = (original_logits - routed_logits).abs()

    trainable = {name: parameter.numel() for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected_trainable = 2 * feature_dim * protocol["architecture"]["rank"]
    if sum(trainable.values()) != expected_trainable or any("adapter" not in name for name in trainable):
        raise AssertionError({"trainable": trainable, "expected": expected_trainable})

    cache = torch.load(args.embeddings, map_location="cpu", weights_only=True)
    cache_index = {example_id: index for index, example_id in enumerate(cache["example_ids"])}
    rows = list(read_jsonl(args.internal_data))
    exact_orders = 0
    metrics = []
    with torch.no_grad():
        for number, source in enumerate(rows, 1):
            index = cache_index[source["example_id"]]
            packets = cache["packets"][index].to(device=device, dtype=torch.float32)
            question = cache["questions"][index].to(device=device, dtype=torch.float32)
            fractions = torch.tensor(source["packet_tokens"], device=device, dtype=torch.float32)
            fractions /= fractions.sum()
            adapted_order = model.beam_order_v17(
                packets, question, fractions, active_level_count=len(trajectory_levels(source)), beam_width=8,
                force_primary_090_active=True,
            )
            model.viability_head = original_head
            original_order = model.beam_order_v17(
                packets, question, fractions, active_level_count=len(trajectory_levels(source)), beam_width=8,
                force_primary_090_active=True,
            )
            model.viability_head = wrapped
            exact_orders += int(adapted_order == original_order)
            exact = list(read_jsonl(Path(args.exact_dir) / f"{source['example_id']}.jsonl", ExactSearchResult))
            metrics.append(metric_for_order(source, exact, list(adapted_order)))
            if number % 50 == 0:
                print(json.dumps({"processed": number, "total": len(rows)}), flush=True)

    levels = sorted({level for row in metrics for level in row["per_level"]}, key=float)
    result = {
        "complete": True,
        "architecture": {
            "feature_dim": feature_dim,
            "rank": protocol["architecture"]["rank"],
            "adapter_parameters": sum(trainable.values()),
            "trainable_parameters": trainable,
            "other_anchor_max_abs_logit_difference": float(logit_diff[..., [0, 1, 2, 4]].max()),
            "primary_max_abs_logit_difference_at_step0": float(logit_diff[..., 3].max()),
            "residual_max_abs_difference_at_step0": float((original_residual-routed_residual).abs().max()),
        },
        "canonical_internal": {
            "examples": len(rows),
            "exact_beam_order_matches": exact_orders,
            "per_level_successes": {level: sum(row["per_level"].get(level, False) for row in metrics) for level in levels},
            "complete_successes": sum(row["complete"] for row in metrics),
        },
        "sealed": {"training": True, "development": True, "confirmation": True, "hop2": True, "terminal_reranker": True},
        "artifacts": {
            "protocol_sha256": sha256(args.protocol_config), "internal_sha256": sha256(args.internal_data),
            "embeddings_sha256": sha256(args.embeddings), "viability_head_sha256": sha256(args.viability_head),
        },
    }
    expected = protocol["architecture_audit_gate"]
    passed = (
        result["architecture"]["other_anchor_max_abs_logit_difference"] == 0.0
        and result["architecture"]["primary_max_abs_logit_difference_at_step0"] == 0.0
        and result["architecture"]["residual_max_abs_difference_at_step0"] == 0.0
        and exact_orders == len(rows)
        and result["canonical_internal"]["per_level_successes"] == expected["per_level_successes"]
        and result["canonical_internal"]["complete_successes"] == expected["complete_successes"]
    )
    result["gate_passed"] = passed
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    write_metadata(out / "architecture_audit.json", result)
    decision = {
        "decision": "GO_V17D2B_ADAPTER_TRAINING" if passed else "STOP_V17D2_ADAPTER_PROTOCOL",
        "training_started": False, "gate_passed": passed,
        "development_used": False, "confirmation_used": False,
    }
    write_metadata(out / "decision.json", decision)
    print(json.dumps({**decision, "audit": result}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
