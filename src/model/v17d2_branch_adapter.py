from __future__ import annotations

import math

import torch
from torch import nn


PRIMARY_090_INDEX = 3


class RankResidualAdapter(nn.Module):
    """Zero-initialized low-rank correction of a frozen action representation."""

    def __init__(self, feature_dim: int, rank: int = 8, layer_norm_eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim, eps=layer_norm_eps, elementwise_affine=False)
        self.down = nn.Linear(feature_dim, rank, bias=False)
        self.up = nn.Linear(rank, feature_dim, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=1.0 / math.sqrt(feature_dim))
        nn.init.zeros_(self.up.weight)

    def correction(self, features: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(self.norm(features)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.correction(features)


class Branch090AdaptedViabilityHead(nn.Module):
    """Route only the 0.90 logit through an adapter; preserve all other logits."""

    def __init__(self, frozen_head: nn.Module, feature_dim: int, rank: int = 8, layer_norm_eps: float = 1e-5):
        super().__init__()
        self.frozen_head = frozen_head
        self.adapter = RankResidualAdapter(feature_dim, rank, layer_norm_eps)
        for parameter in self.frozen_head.parameters():
            parameter.requires_grad = False

    @property
    def residual_scale(self) -> torch.Tensor:
        return self.frozen_head.residual_scale

    def forward(
        self, action_features: torch.Tensor, active_target_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original_logits = self.frozen_head.viability(action_features)
        adapted_logits = self.frozen_head.viability(self.adapter(action_features))
        logits = original_logits.clone()
        logits[..., PRIMARY_090_INDEX] = adapted_logits[..., PRIMARY_090_INDEX]
        weights = active_target_mask.to(logits.dtype)
        denominator = weights.sum(dim=-1).clamp_min(1.0)
        residual = self.residual_scale * (logits * weights).sum(dim=-1) / denominator
        return logits, residual

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.adapter.parameters())
