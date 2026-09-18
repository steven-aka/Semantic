from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F


WeightedEdge = tuple[int, int, float]


@dataclass
class TailCostLoss:
    total: torch.Tensor
    safety_tail: torch.Tensor | None
    rate: torch.Tensor | None
    supervised_examples: int


def tail_cost_relational_loss(
    logits: torch.Tensor,
    safety_edges: Sequence[Sequence[WeightedEdge]],
    rate_edges: Sequence[Sequence[WeightedEdge]],
    *,
    tail_fraction: float = 0.25,
    rate_lambda: float = 0.25,
) -> TailCostLoss:
    """Example-balanced CVaR safety loss plus cost-sensitive rate loss."""
    if logits.ndim != 3 or logits.shape[1] != logits.shape[2]:
        raise ValueError("precedence logits must be a square batch tensor")
    if len(safety_edges) != logits.shape[0] or len(rate_edges) != logits.shape[0]:
        raise ValueError("edge batches must align with logits")
    if not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must be in (0, 1]")
    if rate_lambda < 0:
        raise ValueError("rate_lambda must be non-negative")

    totals = []
    safety_values = []
    rate_values = []
    for batch_index, (example_safety, example_rate) in enumerate(
        zip(safety_edges, rate_edges)
    ):
        safety_loss = None
        if example_safety:
            edge_losses = []
            for winner, loser, weight in example_safety:
                if winner == loser or not 0 <= winner < logits.shape[1] or not 0 <= loser < logits.shape[1]:
                    raise ValueError("invalid safety edge")
                if weight <= 0:
                    raise ValueError("safety edge weight must be positive")
                edge_losses.append(float(weight) * F.softplus(-logits[batch_index, winner, loser]))
            values = torch.stack(edge_losses)
            count = max(1, math.ceil(tail_fraction * len(edge_losses)))
            safety_loss = torch.topk(values, count, largest=True).values.mean()
            safety_values.append(safety_loss)

        rate_loss = None
        if example_rate:
            edge_losses = []
            for winner, loser, weight in example_rate:
                if winner == loser or not 0 <= winner < logits.shape[1] or not 0 <= loser < logits.shape[1]:
                    raise ValueError("invalid rate edge")
                if weight <= 0:
                    raise ValueError("rate edge weight must be positive")
                edge_losses.append(float(weight) * F.softplus(-logits[batch_index, winner, loser]))
            rate_loss = torch.stack(edge_losses).mean()
            rate_values.append(rate_loss)

        if safety_loss is None and rate_loss is None:
            continue
        total = logits.new_zeros(())
        if safety_loss is not None:
            total = total + safety_loss
        if rate_loss is not None:
            total = total + rate_lambda * rate_loss
        totals.append(total)

    if not totals:
        raise ValueError("batch contains no safety or rate supervision")
    return TailCostLoss(
        total=torch.stack(totals).mean(),
        safety_tail=torch.stack(safety_values).mean() if safety_values else None,
        rate=torch.stack(rate_values).mean() if rate_values else None,
        supervised_examples=len(totals),
    )
