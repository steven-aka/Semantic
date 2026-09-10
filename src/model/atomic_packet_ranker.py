from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def partial_order_rank_loss(
    scores: torch.Tensor,
    preferences: Sequence[Sequence[tuple[int, int]]],
) -> torch.Tensor:
    if scores.ndim != 2 or len(preferences) != scores.shape[0]:
        raise ValueError("scores and preference batches must align")
    losses = []
    for batch_index, pairs in enumerate(preferences):
        for winner, loser in pairs:
            if not 0 <= winner < scores.shape[1] or not 0 <= loser < scores.shape[1]:
                raise ValueError("ranking preference index outside packet width")
            losses.append(F.softplus(-(scores[batch_index, winner] - scores[batch_index, loser])))
    if not losses:
        raise ValueError("ranking batch contains no identifiable preference pairs")
    return torch.stack(losses).mean()


class AtomicPacketRanker(nn.Module):
    """Query-conditioned shared packet scorer trained as a partial order."""

    def __init__(self, lm: nn.Module, head_hidden_size: int = 256):
        super().__init__()
        self.lm = lm
        hidden_size = int(lm.config.hidden_size)
        self.rank_head = nn.Sequential(
            nn.Linear(2 * hidden_size, head_hidden_size),
            nn.GELU(),
            nn.Linear(head_hidden_size, 1),
        )

    def predict_scores(
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
            raise ValueError("all rows must contain the same packet count")
        rows = []
        for batch_index, spans in enumerate(packet_spans):
            final_index = int(attention_mask[batch_index].sum().item()) - 1
            global_hidden = hidden[batch_index, final_index]
            features = []
            for start, end in spans:
                if not 0 <= start < end <= final_index:
                    raise ValueError("packet span outside model input")
                local_hidden = hidden[batch_index, start:end].mean(dim=0)
                features.append(torch.cat((local_hidden, global_hidden), dim=-1))
            rows.append(torch.stack(features))
        values = torch.stack(rows).to(self.rank_head[0].weight.dtype)
        return self.rank_head(values).squeeze(-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        packet_spans: Sequence[Sequence[tuple[int, int]]],
        preferences: Sequence[Sequence[tuple[int, int]]] | None = None,
    ) -> dict[str, torch.Tensor]:
        scores = self.predict_scores(input_ids, attention_mask, packet_spans)
        result = {"scores": scores}
        if preferences is not None:
            result["loss"] = partial_order_rank_loss(scores, preferences)
        return result
