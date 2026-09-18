from __future__ import annotations

import torch

from src.model.sequential_packet_policy import SequentialPacketPolicy
from src.training.v17b_viability import ViabilityResidualHead


class V17BViabilityPolicy(SequentialPacketPolicy):
    """V8 policy plus the frozen-protocol multi-anchor residual used by V17-B1."""

    def __init__(self, *args, viability_hidden_dim: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self.viability_head = ViabilityResidualHead(7 * self.model_dim + 66, viability_hidden_dim)

    def score_states_with_viability(
        self,
        packets: torch.Tensor,
        question: torch.Tensor,
        history_state: torch.Tensor,
        selected_mask: torch.Tensor,
        example_indices: torch.Tensor,
        packet_token_fractions: torch.Tensor,
        active_target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, progress = self.action_features(
            packets, question, history_state, selected_mask, example_indices, packet_token_fractions
        )
        base = self.action_head(features).squeeze(-1)
        viability, residual = self.viability_head(features, active_target_mask)
        scores = (base + residual).masked_fill(selected_mask, torch.finfo(base.dtype).min)
        return scores, progress, viability

    @staticmethod
    def deployment_active_target_mask(
        progress_logits: torch.Tensor,
        active_level_counts: torch.Tensor,
        action_count: int = 12,
    ) -> torch.Tensor:
        """Frozen deployable mask: contract metadata plus V8's progress prediction."""
        anchors = torch.arange(5, device=progress_logits.device)[None, :]
        attainable = anchors < active_level_counts[:, None]
        predicted_resolved = progress_logits.sigmoid() >= 0.5
        return (attainable & ~predicted_resolved)[:, None, :].expand(-1, action_count, -1)

    def freeze_v8(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith("viability_head.")
