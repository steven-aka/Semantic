from __future__ import annotations

import torch
from torch import nn


class MaskValueHead(nn.Module):
    """Predict ordinal active-anchor attainment for an arbitrary packet mask."""

    def __init__(
        self,
        *,
        model_dim: int = 512,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.selection = nn.Embedding(2, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=2 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mask_encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.output = nn.Sequential(
            nn.Linear(4 * model_dim + 1, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 5),
        )

    def forward(
        self,
        packets: torch.Tensor,
        question: torch.Tensor,
        masks: torch.Tensor,
        example_indices: torch.Tensor,
    ) -> torch.Tensor:
        if packets.ndim != 3 or packets.shape[1:] != (12, self.model_dim):
            raise ValueError("packets must have shape [examples, 12, model_dim]")
        if question.ndim != 2 or question.shape[1] != self.model_dim:
            raise ValueError("question must have shape [examples, model_dim]")
        if masks.ndim != 1:
            raise ValueError("masks must be a flat integer tensor")
        if example_indices.shape != masks.shape:
            raise ValueError("example indices must align with masks")
        if bool(((masks < 0) | (masks >= 4096)).any()):
            raise ValueError("masks must be in [0, 4095]")
        selected = (
            masks[:, None].bitwise_and(1 << torch.arange(12, device=masks.device)[None, :]) != 0
        )
        values = packets[example_indices] + self.selection(selected.long())
        values = self.mask_encoder(values)
        selected_float = selected.to(values.dtype)
        remaining_float = (~selected).to(values.dtype)
        selected_count = selected_float.sum(dim=1, keepdim=True)
        remaining_count = remaining_float.sum(dim=1, keepdim=True)
        selected_pool = (values * selected_float[:, :, None]).sum(dim=1) / selected_count.clamp_min(1)
        remaining_pool = (values * remaining_float[:, :, None]).sum(dim=1) / remaining_count.clamp_min(1)
        selected_pool = torch.where(selected_count > 0, selected_pool, torch.zeros_like(selected_pool))
        remaining_pool = torch.where(remaining_count > 0, remaining_pool, torch.zeros_like(remaining_pool))
        query = question[example_indices]
        fraction = selected_count.to(values.dtype) / 12.0
        raw = self.output(
            torch.cat((query, selected_pool, remaining_pool, selected_pool * remaining_pool, fraction), dim=-1)
        )
        first = raw[:, :1]
        decrements = torch.nn.functional.softplus(raw[:, 1:])
        return torch.cat((first, first - torch.cumsum(decrements, dim=1)), dim=1)


def ordinal_mask_value_loss(
    logits: torch.Tensor,
    attained_levels: torch.Tensor,
    sampling_probabilities: torch.Tensor,
    example_indices: torch.Tensor,
) -> torch.Tensor:
    anchors = torch.arange(5, device=logits.device)[None, :]
    target = anchors < attained_levels[:, None]
    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.float(), target.float(), reduction="none"
    )
    if sampling_probabilities.shape != attained_levels.shape or example_indices.shape != attained_levels.shape:
        raise ValueError("sampling probabilities and example indices must align with labels")
    if bool(((sampling_probabilities <= 0) | (sampling_probabilities > 1)).any()):
        raise ValueError("sampling probabilities must be in (0, 1]")
    weights = sampling_probabilities.float().reciprocal()
    losses = []
    for example in torch.unique(example_indices, sorted=True):
        selected = example_indices == example
        denominator = weights[selected].sum()
        losses.append((raw[selected] * weights[selected, None]).sum(dim=0).div(denominator).mean())
    if not losses:
        raise ValueError("ordinal mask-value loss requires at least one example")
    return torch.stack(losses).mean()
