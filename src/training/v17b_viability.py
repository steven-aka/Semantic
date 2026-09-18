from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from random import Random
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn


FIDELITY_GRID = (0.60, 0.70, 0.80, 0.90, 0.95)


class ViabilityResidualHead(nn.Module):
    """Small V17-B head; its zero-initialized residual preserves V8 at step zero."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.viability = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(FIDELITY_GRID))
        )
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, action_features: torch.Tensor, active_target_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.viability(action_features)
        weights = active_target_mask.to(logits.dtype)
        denominator = weights.sum(dim=-1).clamp_min(1.0)
        residual = self.residual_scale * (logits * weights).sum(dim=-1) / denominator
        return logits, residual


def masked_viability_loss(logits: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    weights = target_mask.to(losses.dtype)
    # Macro-average anchors so a five-anchor example cannot dominate a four-anchor example.
    per_anchor = (losses * weights).sum(dim=(0, 1)) / weights.sum(dim=(0, 1)).clamp_min(1.0)
    active = weights.sum(dim=(0, 1)) > 0
    if not bool(active.any()):
        raise ValueError("batch has no active viability targets")
    return per_anchor[active].mean()


def set_valued_action_loss(action_logits: torch.Tensor, optimal_mask: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    masked = action_logits.float().masked_fill(~legal_mask, -torch.inf)
    numerator = torch.logsumexp(masked.masked_fill(~optimal_mask, -torch.inf), dim=-1)
    denominator = torch.logsumexp(masked, dim=-1)
    if not bool(torch.isfinite(numerator).all()):
        raise ValueError("every state must have at least one legal exact-DP optimal action")
    return (denominator - numerator).mean()


def viability_mass_loss(
    action_logits: torch.Tensor,
    viability_targets: torch.Tensor,
    target_mask: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Local beam-survival surrogate: retain probability mass on every viable action set."""
    log_probs = torch.log_softmax(action_logits.float().masked_fill(~legal_mask, -torch.inf), dim=1)
    losses = []
    for state in range(action_logits.shape[0]):
        for anchor in range(viability_targets.shape[-1]):
            active = target_mask[state, :, anchor] & legal_mask[state]
            positive = active & viability_targets[state, :, anchor].bool()
            if bool(active.any()) and bool(positive.any()):
                losses.append(-torch.logsumexp(log_probs[state].masked_fill(~positive, -torch.inf), dim=0))
    if not losses:
        raise ValueError("batch has no viable active-anchor action set")
    return torch.stack(losses).mean()


@dataclass(frozen=True)
class StratifiedStateSampler:
    """Query-balanced 50/50 deployed/one-hop sampling frozen for V17-B1."""

    records: Sequence[dict[str, Any]]
    seed: int

    def sample(self, batch_size: int, step: int) -> list[int]:
        if batch_size < 2 or batch_size % 2:
            raise ValueError("V17-B batches must have an even size of at least two")
        rng = Random(self.seed + step)
        by_layer: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, row in enumerate(self.records):
            by_layer[row["provenance"]][row["example_id"]].append(index)
        output: list[int] = []
        for layer in ("deployed", "one_hop"):
            queries = sorted(by_layer[layer])
            if not queries:
                raise ValueError(f"missing sampler layer: {layer}")
            for _ in range(batch_size // 2):
                query = rng.choice(queries)
                candidates = by_layer[layer][query]
                # State depth and viability pattern remain explicit strata in the artifact;
                # query balancing is the outer sampling unit and prevents prolific queries dominating.
                output.append(rng.choice(candidates))
        rng.shuffle(output)
        return output
