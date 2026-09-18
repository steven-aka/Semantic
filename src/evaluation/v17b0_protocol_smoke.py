from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import torch

from src.reproducibility import sha256, write_metadata
from src.training.v17b_viability import StratifiedStateSampler, ViabilityResidualHead, masked_viability_loss, set_valued_action_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Engineering-only smoke test for frozen V17-B0 protocol")
    parser.add_argument("--artifact", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=40); parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args(); torch.manual_seed(args.seed)
    with gzip.open(args.artifact, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    sampler = StratifiedStateSampler(records, args.seed)
    sampled = sampler.sample(32, 0)
    layer_counts = {layer: sum(records[i]["provenance"] == layer for i in sampled) for layer in ("deployed", "one_hop")}

    # Synthetic features isolate the trainable loss/head contract from the frozen 4B encoder.
    head = ViabilityResidualHead(16, 16)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.02)
    features = torch.randn(8, 12, 16)
    targets = torch.zeros(8, 12, 5)
    masks = torch.zeros(8, 12, 5, dtype=torch.bool)
    for anchor in range(5):
        masks[anchor, :, anchor] = True
        targets[anchor, :6, anchor] = 1
    initial = final = None; anchor_gradient = None
    for step in range(args.steps):
        logits, residual = head(features, masks)
        viability = masked_viability_loss(logits, targets, masks)
        action_logits = residual
        legal = torch.ones_like(action_logits, dtype=torch.bool)
        optimal = torch.zeros_like(legal); optimal[:, :6] = True
        set_loss = set_valued_action_loss(action_logits, optimal, legal)
        loss = viability + .1 * set_loss
        optimizer.zero_grad(); loss.backward()
        if step == 0:
            initial = float(loss.detach())
            anchor_gradient = [float(head.viability[-1].weight.grad[a].abs().sum()) for a in range(5)]
        optimizer.step(); final = float(loss.detach())
    with torch.no_grad():
        fresh = ViabilityResidualHead(16, 16)
        _, zero_residual = fresh(features, masks)
    result = {
        "complete": True, "engineering_only": True, "development_used": False,
        "artifact": args.artifact, "artifact_sha256": sha256(args.artifact),
        "sampler_batch_layer_counts": layer_counts, "loss_initial": initial, "loss_final": final,
        "loss_decreased": final < initial, "active_anchor_output_gradient_l1_at_step0": anchor_gradient,
        "all_active_anchors_receive_gradient": all(value > 0 for value in anchor_gradient),
        "set_valued_loss_finite": True, "fresh_zero_residual_max_abs": float(zero_residual.abs().max()),
        "zero_residual_preserves_v8": bool(float(zero_residual.abs().max()) == 0.0),
    }
    if not (result["loss_decreased"] and result["all_active_anchors_receive_gradient"] and result["zero_residual_preserves_v8"] and layer_counts == {"deployed": 16, "one_hop": 16}):
        raise RuntimeError(result)
    write_metadata(args.output, result); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
