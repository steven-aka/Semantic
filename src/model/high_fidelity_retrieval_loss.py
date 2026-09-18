from __future__ import annotations
import torch

def high_fidelity_retrieval_loss(logits,attained_levels,example_indices,anchor_index=3):
 """Query-balanced multiple-positive softmax loss for retrieving an anchor-feasible mask."""
 scores=logits[:,anchor_index].float();losses=[]
 for e in torch.unique(example_indices,sorted=True):
  sel=example_indices==e;pos=sel & (attained_levels>anchor_index)
  if not bool(pos.any()):raise ValueError('each query requires a positive retrieval mask')
  losses.append(torch.logsumexp(scores[sel],0)-torch.logsumexp(scores[pos],0))
 return torch.stack(losses).mean()
