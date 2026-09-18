from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class CostSensitivePolicyLoss:
    total: torch.Tensor
    action: torch.Tensor
    progress: torch.Tensor
    zero_advantage_accuracy: float
    mean_selected_advantage: float
    mean_expected_advantage: float


def cost_sensitive_policy_loss(
    action_logits: torch.Tensor,
    progress_logits: torch.Tensor,
    action_advantages: list[list[float | None]],
    reached_levels: torch.Tensor,
    active_level_counts: torch.Tensor,
    *,
    progress_weight: float = 0.2,
) -> CostSensitivePolicyLoss:
    """Cost-weighted ranking loss over exact downstream action advantages.

    Every positive-cost action is compared with the aggregate score of all
    zero-cost actions. Its exact cost-to-go advantage scales the penalty, so a
    lost reachable anchor contributes more than a small token inefficiency.
    """
    if action_logits.ndim != 2 or action_logits.shape[1] != 12:
        raise ValueError("action logits must have shape [states, 12]")
    if progress_logits.shape != (action_logits.shape[0], 5):
        raise ValueError("progress logits must align with action states")
    if len(action_advantages) != action_logits.shape[0]:
        raise ValueError("action advantages must align with logits")
    if any(len(values) != 12 for values in action_advantages):
        raise ValueError("each action-advantage row must have width twelve")
    costs = torch.tensor(
        [[float("nan") if value is None else float(value) for value in row] for row in action_advantages],
        dtype=torch.float32,
        device=action_logits.device,
    )
    remaining = ~torch.isnan(costs)
    optimal = remaining & (costs <= 1e-12)
    harmful = remaining & (costs > 1e-12)
    if not bool(optimal.any(dim=1).all().item()):
        raise ValueError("every nonterminal state needs a zero-advantage action")
    logits = action_logits.float().masked_fill(~remaining, float("-inf"))
    optimal_score = torch.logsumexp(logits.masked_fill(~optimal, float("-inf")), dim=1)
    comparisons = F.softplus(logits - optimal_score[:, None])
    finite_costs = torch.nan_to_num(costs, nan=0.0)
    per_state = (finite_costs * comparisons * harmful).sum(dim=1) / harmful.sum(dim=1).clamp_min(1)
    action = per_state.mean()
    selected = logits.argmax(dim=1)
    selected_advantages = finite_costs.gather(1, selected[:, None]).squeeze(1)
    probabilities = torch.softmax(logits, dim=1)
    expected_advantages = (probabilities * finite_costs).sum(dim=1)
    anchors = torch.arange(5, device=progress_logits.device)[None, :]
    target_progress = anchors < reached_levels[:, None]
    active = anchors < active_level_counts[:, None]
    raw = F.binary_cross_entropy_with_logits(
        progress_logits.float(), target_progress.float(), reduction="none"
    )
    progress = raw[active].mean()
    total = action + progress_weight * progress
    return CostSensitivePolicyLoss(
        total=total,
        action=action,
        progress=progress,
        zero_advantage_accuracy=float((selected_advantages <= 1e-12).float().mean().detach().item()),
        mean_selected_advantage=float(selected_advantages.mean().detach().item()),
        mean_expected_advantage=float(expected_advantages.mean().detach().item()),
    )
