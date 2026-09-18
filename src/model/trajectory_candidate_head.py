from __future__ import annotations
import torch
from torch import nn
from src.model.mask_value_head import MaskValueHead

class TrajectoryCandidateHead(MaskValueHead):
    """Score a candidate target mask whose projection defines a full packet order."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output = nn.Sequential(
            nn.Linear(4 * self.model_dim + 1, self.model_dim), nn.GELU(), nn.Dropout(kwargs.get('dropout', .1)), nn.Linear(self.model_dim, 1)
        )
    def forward(self, packets, question, masks, example_indices):
        # Reuse the mask-conditioned representation, with an unconstrained scalar output.
        if packets.ndim != 3 or packets.shape[1:] != (12, self.model_dim): raise ValueError('invalid packets')
        selected=(masks[:,None].bitwise_and(1<<torch.arange(12,device=masks.device)[None,:])!=0)
        values=self.mask_encoder(packets[example_indices]+self.selection(selected.long()))
        sf=selected.to(values.dtype); rf=(~selected).to(values.dtype)
        sc=sf.sum(1,keepdim=True); rc=rf.sum(1,keepdim=True)
        sp=(values*sf[:,:,None]).sum(1)/sc.clamp_min(1); rp=(values*rf[:,:,None]).sum(1)/rc.clamp_min(1)
        sp=torch.where(sc>0,sp,torch.zeros_like(sp));rp=torch.where(rc>0,rp,torch.zeros_like(rp))
        q=question[example_indices]
        return self.output(torch.cat((q,sp,rp,sp*rp,sc.to(values.dtype)/12),-1)).squeeze(-1)

def candidate_set_loss(logits, optimal, example_indices):
    losses=[]
    for e in torch.unique(example_indices,sorted=True):
        sel=example_indices==e
        if not bool(optimal[sel].any()): raise ValueError('each query needs an optimal candidate')
        losses.append(torch.logsumexp(logits[sel].float(),0)-torch.logsumexp(logits[sel].float().masked_fill(~optimal[sel],float('-inf')),0))
    return torch.stack(losses).mean()
