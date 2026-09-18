from __future__ import annotations
import torch
from torch import nn


def pool(packets, masks, example_indices):
    selected=masks[:,None].bitwise_and(1<<torch.arange(12,device=masks.device)[None])!=0;weight=selected.to(packets.dtype);count=weight.sum(1,keepdim=True);values=(packets[example_indices]*weight[:,:,None]).sum(1)/count.clamp_min(1);return torch.where(count>0,values,torch.zeros_like(values))


class CounterfactualRepairVerifier(nn.Module):
    def __init__(self,dim=512,dropout=.1):
        super().__init__();self.rank=nn.Embedding(5,32);width=8*dim+32+6
        self.body=nn.Sequential(nn.Linear(width,dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim,dim),nn.GELU(),nn.Dropout(dropout));self.classes=nn.Linear(dim,4);self.delta=nn.Linear(dim,1)
    def forward(self,packets,questions,candidate,matched,added,removed,ranks,retrieval,fractions,indices):
        cp=pool(packets,candidate,indices);bp=pool(packets,matched,indices);ap=pool(packets,added,indices);rp=pool(packets,removed,indices);q=questions[indices]
        h=self.body(torch.cat((q,cp,bp,ap,rp,cp-bp,ap-rp,cp*bp,self.rank(ranks),retrieval.to(cp.dtype),fractions[:,None].to(cp.dtype)),1));return self.classes(h),self.delta(h).squeeze(1)


def verifier_loss(class_logits,delta_pred,class_target,delta_target,example_indices,class_weights):
    ce=torch.nn.functional.cross_entropy(class_logits.float(),class_target,reduction="none",weight=class_weights);reg=torch.nn.functional.smooth_l1_loss(delta_pred.float(),delta_target.float(),reduction="none");losses=[]
    for e in torch.unique(example_indices,sorted=True):
        selected=example_indices==e;losses.append(ce[selected].mean()+.5*reg[selected].mean())
    return torch.stack(losses).mean()
