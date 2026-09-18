from __future__ import annotations

from typing import Sequence

import torch
from torch.nn import functional as F


def worst_boundary_rank_loss(
    scores: torch.Tensor,
    preferences: Sequence[Sequence[tuple[int, int]]],
) -> torch.Tensor:
    """Example-balanced softplus loss on each example's worst boundary pair."""
    if scores.ndim != 2 or len(preferences) != scores.shape[0]:
        raise ValueError("scores and preference batches must align")
    losses = []
    for batch_index, pairs in enumerate(preferences):
        violations = []
        for winner, loser in pairs:
            if not 0 <= winner < scores.shape[1] or not 0 <= loser < scores.shape[1]:
                raise ValueError("boundary preference index outside packet width")
            violations.append(scores[batch_index, loser] - scores[batch_index, winner])
        if violations:
            losses.append(F.softplus(torch.stack(violations).max()))
    if not losses:
        raise ValueError("ranking batch contains no critical-boundary pairs")
    return torch.stack(losses).mean()
