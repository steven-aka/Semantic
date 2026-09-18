from __future__ import annotations

import torch
from torch import nn


class EvidenceSufficiencySelector(nn.Module):
    """Predict five-anchor trajectory sufficiency for a frozen candidate order."""

    def __init__(self, model_dim: int = 512, layers: int = 2, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.model_dim = model_dim
        self.selection = nn.Embedding(2, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=heads, dim_feedforward=2 * model_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.mask_encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.rank_embedding = nn.Embedding(5, 32)
        self.output = nn.Sequential(
            nn.Linear(4 * model_dim + 1 + 5 + 1 + 32, model_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(model_dim, 5),
        )

    def forward(self, packets, question, masks, ranks, retrieval_logits, token_fractions, example_indices):
        selected = masks[:, None].bitwise_and(1 << torch.arange(12, device=masks.device)[None]) != 0
        values = self.mask_encoder(packets[example_indices] + self.selection(selected.long()))
        sf = selected.to(values.dtype); rf = (~selected).to(values.dtype)
        sc = sf.sum(1, keepdim=True); rc = rf.sum(1, keepdim=True)
        sp = (values * sf[:, :, None]).sum(1) / sc.clamp_min(1)
        rp = (values * rf[:, :, None]).sum(1) / rc.clamp_min(1)
        sp = torch.where(sc > 0, sp, torch.zeros_like(sp))
        rp = torch.where(rc > 0, rp, torch.zeros_like(rp))
        q = question[example_indices]
        features = torch.cat((q, sp, rp, sp * rp, sc.to(values.dtype) / 12,
                              retrieval_logits.to(values.dtype), token_fractions[:, None].to(values.dtype),
                              self.rank_embedding(ranks).to(values.dtype)), dim=-1)
        raw = self.output(features)
        first = raw[:, :1]
        return torch.cat((first, first - torch.cumsum(torch.nn.functional.softplus(raw[:, 1:]), 1)), 1)


def conservative_sufficiency_loss(logits, targets, active, example_indices, harm_weight: float = 4.0):
    raw = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    weights = torch.tensor([1.0, 1.0, 1.0, 2.0, 1.0], device=logits.device)
    losses = []
    for example in torch.unique(example_indices, sorted=True):
        selected = example_indices == example
        base_target = targets[selected][0]
        per_candidate = (raw[selected] * active[selected].float() * weights).sum(1) / (active[selected].float() * weights).sum(1).clamp_min(1)
        score = (torch.sigmoid(logits[selected].float()) * active[selected].float() * weights).sum(1)
        utility = (targets[selected].float() * active[selected].float() * weights).sum(1)
        delta = utility[1:] - utility[0]
        pair = torch.zeros_like(delta)
        pair = torch.where(delta > 0, torch.nn.functional.softplus(-(score[1:] - score[0])), pair)
        pair = torch.where(delta < 0, harm_weight * torch.nn.functional.softplus(score[1:] - score[0] + 1.0), pair)
        losses.append(per_candidate.mean() + pair.mean())
    return torch.stack(losses).mean()
