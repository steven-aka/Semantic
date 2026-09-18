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

    def beam_order_v17(
        self,
        packets: torch.Tensor,
        question: torch.Tensor,
        packet_token_fractions: torch.Tensor,
        *,
        active_level_count: int,
        beam_width: int = 8,
        force_primary_090_active: bool = False,
    ) -> tuple[int, ...]:
        dtype = self.initial_history.weight.dtype
        packets, question = packets.to(dtype), question.to(dtype)
        beams = [(0.0, (), torch.tanh(self.initial_history(question[None]))[0])]
        for _ in range(12):
            count = len(beams); states = torch.stack([row[2] for row in beams])
            selected = torch.zeros((count, 12), dtype=torch.bool, device=packets.device)
            for index, (_, history, _) in enumerate(beams):
                if history: selected[index, list(history)] = True
            expanded_packets = packets[None].expand(count, -1, -1)
            expanded_questions = question[None].expand(count, -1)
            indices = torch.arange(count, device=packets.device)
            fractions = packet_token_fractions[None].expand(count, -1)
            features, progress = self.action_features(expanded_packets, expanded_questions, states, selected, indices, fractions)
            active = self.deployment_active_target_mask(progress, torch.full((count,), active_level_count, device=packets.device))
            if force_primary_090_active:
                active = active.clone()
                active[:, :, 3] = True
            base = self.action_head(features).squeeze(-1)
            _, residual = self.viability_head(features, active)
            logits = (base + residual).masked_fill(selected, torch.finfo(base.dtype).min)
            log_prob = torch.log_softmax(logits.float(), dim=-1).cpu()
            next_states = self.history_gru(
                expanded_packets.reshape(-1, self.model_dim),
                states[:, None, :].expand(-1, 12, -1).reshape(-1, self.model_dim),
            ).reshape(count, 12, self.model_dim)
            expanded = []
            for index, (score, history, _) in enumerate(beams):
                for packet in range(12):
                    if not selected[index, packet]:
                        expanded.append((score + float(log_prob[index, packet]), history + (packet,), next_states[index, packet]))
            expanded.sort(key=lambda row: (-row[0], row[1])); beams = expanded[:beam_width]
        return beams[0][1]
