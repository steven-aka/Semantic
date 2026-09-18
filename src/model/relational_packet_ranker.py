from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class RelationalPacketRanker(nn.Module):
    """One-backbone-forward antisymmetric packet-precedence scorer."""

    def __init__(self, lm: nn.Module, head_hidden_size: int = 256):
        super().__init__()
        self.lm = lm
        hidden_size = int(lm.config.hidden_size)
        self.precedence_head = nn.Sequential(
            nn.Linear(5 * hidden_size, head_hidden_size),
            nn.GELU(),
            nn.Linear(head_hidden_size, 1),
        )

    def predict_precedence_logits(
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
        local_rows = []
        global_rows = []
        for batch_index, spans in enumerate(packet_spans):
            final_index = int(attention_mask[batch_index].sum().item()) - 1
            global_rows.append(hidden[batch_index, final_index])
            local = []
            for start, end in spans:
                if not 0 <= start < end <= final_index:
                    raise ValueError("packet span outside model input")
                local.append(hidden[batch_index, start:end].mean(dim=0))
            local_rows.append(torch.stack(local))
        local_hidden = torch.stack(local_rows)
        global_hidden = torch.stack(global_rows)
        width = local_hidden.shape[1]
        first = local_hidden[:, :, None, :].expand(-1, -1, width, -1)
        second = local_hidden[:, None, :, :].expand(-1, width, -1, -1)
        global_pair = global_hidden[:, None, None, :].expand(-1, width, width, -1)
        features = torch.cat(
            (first, second, first - second, first * second, global_pair), dim=-1
        ).to(self.precedence_head[0].weight.dtype)
        directed = self.precedence_head(features).squeeze(-1)
        logits = directed - directed.transpose(1, 2)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        packet_spans: Sequence[Sequence[tuple[int, int]]],
    ) -> dict[str, torch.Tensor]:
        return {
            "precedence_logits": self.predict_precedence_logits(
                input_ids, attention_mask, packet_spans
            )
        }
