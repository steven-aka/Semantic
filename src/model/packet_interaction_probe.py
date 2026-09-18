from __future__ import annotations

import torch
from torch import nn


class PacketInteractionProbe(nn.Module):
    """Diagnostic probe that preserves packet identity and candidate/baseline interactions."""

    def __init__(self, dim: int = 512, layers: int = 3, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.membership = nn.Embedding(4, dim)
        self.position = nn.Embedding(13, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, "gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.rank = nn.Embedding(5, 32)
        self.output = nn.Sequential(nn.Linear(3 * dim + 32 + 6, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 4 + 15))

    def forward(self, packets, questions, candidate, matched, added, removed, ranks, retrieval, fractions, indices):
        device=packets.device; bits=1 << torch.arange(12,device=device)[None]
        cand=(candidate[:,None]&bits)!=0; base=(matched[:,None]&bits)!=0
        membership=cand.long()+2*base.long()
        q=questions[indices]; p=packets[indices]
        tokens=torch.cat((q[:,None],p),1)
        status=torch.cat((torch.zeros((len(indices),1),dtype=torch.long,device=device),membership),1)
        tokens=tokens+self.membership(status)+self.position(torch.arange(13,device=device))[None]
        encoded=self.encoder(tokens); added_flag=(added[:,None]&bits)!=0; removed_flag=(removed[:,None]&bits)!=0
        def mean(flag):
            weight=flag.to(encoded.dtype); count=weight.sum(1,keepdim=True)
            value=(encoded[:,1:]*weight[:,:,None]).sum(1)/count.clamp_min(1)
            return torch.where(count>0,value,torch.zeros_like(value))
        feature=torch.cat((encoded[:,0],mean(added_flag),mean(removed_flag),self.rank(ranks),retrieval.to(encoded.dtype),fractions[:,None].to(encoded.dtype)),1)
        raw=self.output(feature)
        return raw[:,:4],raw[:,4:].reshape(-1,5,3)


def interaction_loss(class_logits,anchor_logits,class_target,anchor_target,active,example_indices,class_weights):
    class_loss=torch.nn.functional.cross_entropy(class_logits.float(),class_target,reduction="none",weight=class_weights)
    anchor_loss=torch.nn.functional.cross_entropy(anchor_logits.float().reshape(-1,3),anchor_target.reshape(-1),reduction="none").reshape(-1,5)
    anchor_loss=(anchor_loss*active.float()).sum(1)/active.float().sum(1).clamp_min(1)
    losses=[]
    for example in torch.unique(example_indices,sorted=True):
        selected=example_indices==example; losses.append(class_loss[selected].mean()+0.5*anchor_loss[selected].mean())
    return torch.stack(losses).mean()
