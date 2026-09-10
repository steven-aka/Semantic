from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def trajectory_logits(
    thresholds: torch.Tensor, levels: torch.Tensor, temperature: float
) -> torch.Tensor:
    if thresholds.ndim != 2 or levels.ndim != 1:
        raise ValueError("thresholds must be [batch, packet] and levels [anchor]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return (levels.view(1, 1, -1) - thresholds.unsqueeze(-1)) / temperature


def masked_ordinal_bce(
    thresholds: torch.Tensor,
    levels: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    logits = trajectory_logits(thresholds, levels, temperature)
    if labels.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError("labels and mask must match [batch, packet, anchor]")
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    weight = mask.to(losses.dtype)
    denominator = weight.sum()
    if denominator.item() == 0:
        raise ValueError("ordinal batch contains no supervised anchors")
    return (losses * weight).sum() / denominator


class AtomicRevealPolicy(nn.Module):
    """One shared reveal-threshold head over query-conditioned atomic packets."""

    def __init__(self, lm: nn.Module):
        super().__init__()
        self.lm = lm
        hidden_size = int(lm.config.hidden_size)
        self.threshold_head = nn.Linear(2 * hidden_size, 1)

    def predict_thresholds(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        packet_spans: Sequence[Sequence[tuple[int, int]]],
    ) -> torch.Tensor:
        output = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = output.hidden_states[-1]
        packet_counts = {len(spans) for spans in packet_spans}
        if len(packet_counts) != 1:
            raise ValueError("all batch rows must have the same packet count")
        rows = []
        for batch_index, spans in enumerate(packet_spans):
            final_index = int(attention_mask[batch_index].sum().item()) - 1
            global_hidden = hidden[batch_index, final_index]
            packet_rows = []
            for start, end in spans:
                if not 0 <= start < end <= final_index:
                    raise ValueError("packet span is outside the unpadded model input")
                local_hidden = hidden[batch_index, start:end].mean(dim=0)
                packet_rows.append(torch.cat((local_hidden, global_hidden), dim=-1))
            rows.append(torch.stack(packet_rows))
        features = torch.stack(rows).to(self.threshold_head.weight.dtype)
        return torch.sigmoid(self.threshold_head(features).squeeze(-1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        packet_spans: Sequence[Sequence[tuple[int, int]]],
        levels: torch.Tensor,
        labels: torch.Tensor | None = None,
        label_mask: torch.Tensor | None = None,
        *,
        temperature: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        thresholds = self.predict_thresholds(input_ids, attention_mask, packet_spans)
        result = {
            "thresholds": thresholds,
            "trajectory_logits": trajectory_logits(thresholds, levels, temperature),
        }
        if labels is not None:
            if label_mask is None:
                raise ValueError("label_mask is required with labels")
            result["loss"] = masked_ordinal_bce(
                thresholds,
                levels,
                labels,
                label_mask,
                temperature=temperature,
            )
        return result
