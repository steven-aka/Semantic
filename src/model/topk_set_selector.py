"""Small candidate-set scorer with a single deployable top-1 decision."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


FEATURE_DIM = 4 * 512 + 9


class TopKSetSelector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Sequential(nn.LayerNorm(FEATURE_DIM), nn.Linear(FEATURE_DIM, 128), nn.GELU())
        layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0.1, batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, num_layers=1)
        self.score = nn.Linear(128, 1)

    def forward(self, features: torch.Tensor, unique: torch.Tensor) -> torch.Tensor:
        values = self.embed(features.float())
        values = self.context(values, src_key_padding_mask=~unique)
        return self.score(values).squeeze(-1).masked_fill(~unique, -torch.inf)


def successful_set_loss(scores: torch.Tensor, success: torch.Tensor) -> torch.Tensor:
    """Negative log mass on the successful set; no label is forced to be unique."""
    eligible = success.any(dim=1)
    if not bool(eligible.any()):
        raise ValueError("batch has no successful candidate sets")
    selected = scores[eligible].float()
    positive = success[eligible]
    log_all = torch.logsumexp(selected, dim=1)
    log_good = torch.logsumexp(selected.masked_fill(~positive, -torch.inf), dim=1)
    return (log_all - log_good).mean()
