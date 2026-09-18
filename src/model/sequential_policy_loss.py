from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class SequentialPolicyLoss:
    total: torch.Tensor
    action: torch.Tensor
    progress: torch.Tensor
    action_accuracy: float


def sequential_policy_loss(
    action_logits: torch.Tensor,
    progress_logits: torch.Tensor,
    optimal_actions: list[list[int]],
    reached_levels: torch.Tensor,
    active_level_counts: torch.Tensor,
    *,
    progress_weight: float = 0.2,
) -> SequentialPolicyLoss:
    if action_logits.ndim != 2 or action_logits.shape[1] != 12:
        raise ValueError("action logits must have shape [states, 12]")
    if progress_logits.shape != (action_logits.shape[0], 5):
        raise ValueError("progress logits must align with action states")
    if len(optimal_actions) != action_logits.shape[0]:
        raise ValueError("optimal action sets must align with logits")
    log_probs = torch.log_softmax(action_logits.float(), dim=-1)
    losses = []
    correct = 0
    for index, actions in enumerate(optimal_actions):
        if not actions:
            raise ValueError("every nonterminal history needs an optimal action")
        target = torch.tensor(actions, dtype=torch.long, device=action_logits.device)
        losses.append(-torch.logsumexp(log_probs[index, target], dim=0))
        correct += int(int(action_logits[index].argmax().item()) in actions)
    action = torch.stack(losses).mean()
    anchors = torch.arange(5, device=progress_logits.device)[None, :]
    target_progress = anchors < reached_levels[:, None]
    active = anchors < active_level_counts[:, None]
    raw = F.binary_cross_entropy_with_logits(
        progress_logits.float(), target_progress.float(), reduction="none"
    )
    progress = raw[active].mean()
    total = action + progress_weight * progress
    return SequentialPolicyLoss(total, action, progress, correct / len(optimal_actions))
