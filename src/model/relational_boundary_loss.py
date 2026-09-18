from __future__ import annotations

from typing import Sequence

import torch
from torch.nn import functional as F


def worst_relational_boundary_loss(
    logits: torch.Tensor,
    preferences: Sequence[Sequence[tuple[int, int]]],
) -> torch.Tensor:
    """Example-balanced worst-edge logistic loss; no margin or auxiliary term."""
    if logits.ndim != 3 or logits.shape[1] != logits.shape[2]:
        raise ValueError("precedence logits must be a square batch tensor")
    if len(preferences) != logits.shape[0]:
        raise ValueError("logits and preference batches must align")
    losses = []
    for batch_index, pairs in enumerate(preferences):
        values = []
        for winner, loser in pairs:
            if not 0 <= winner < logits.shape[1] or not 0 <= loser < logits.shape[1]:
                raise ValueError("boundary preference index outside packet width")
            if winner == loser:
                raise ValueError("self precedence is invalid")
            values.append(-logits[batch_index, winner, loser])
        if values:
            losses.append(F.softplus(torch.stack(values).max()))
    if not losses:
        raise ValueError("ranking batch contains no critical-boundary pairs")
    return torch.stack(losses).mean()
