from __future__ import annotations
import argparse,json,random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.data.schemas import read_jsonl,write_jsonl
from src.model.high_fidelity_retrieval_loss import high_fidelity_retrieval_loss
from src.model.mask_value_head import MaskValueHead
from src.reproducibility import experiment_metadata,sha256,write_metadata
from src.training.mask_value_data import CachedMaskValueCollator,CachedMaskValueDataset

def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol-config',required=True);p.add_argument('--train-data',required=True);p.add_argument('--train-cache',required=True);p.add_argument('--v10-head',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--steps',type=int,default=400);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=20260918);a=p.parse_args();c=json.loads(Path(a.protocol_config).read_text())
 if c['status']!='APPROVED_TO_RUN':raise ValueError('protocol not approved')
 exp={'train_data':c['data']['train'],'train_cache':c['data']['train_cache'],'v10_head':c['frozen']['v10_head'],'output_dir':c['output'],'steps':c['optimization']['steps'],'batch_size':c['optimization']['batch_size'],'lr':c['optimization']['learning_rate'],'seed':c['optimization']['seed']};mis={k:(getattr(a,k),v) for k,v in exp.items() if getattr(a,k)!=v}
 if mis:raise ValueError(mis)
 random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);device=torch.device('cuda:0');out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);write_metadata(out/'protocol_snapshot.json',c)
 rows=list(read_jsonl(a.train_data));ds=CachedMaskValueDataset(rows,a.train_cache);loader=DataLoader(ds,batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed),collate_fn=CachedMaskValueCollator());it=iter(loader)
 model=MaskValueHead(model_dim=512,layers=2,heads=8,dropout=.1);model.load_state_dict(torch.load(a.v10_head,map_location='cpu',weights_only=True));model.to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps);hist=[]
 for step in range(1,a.steps+1):
  model.train()
  try:b=next(it)
  except StopIteration:it=iter(loader);b=next(it)
  with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=model(b['packets'].to(device),b['questions'].to(device),b['masks'].to(device),b['example_indices'].to(device));loss=high_fidelity_retrieval_loss(logits,b['attained_levels'].to(device),b['example_indices'].to(device))
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();sched.step();opt.zero_grad(set_to_none=True)
  if step%20==0:hist.append({'step':step,'retrieval_loss':float(loss.detach())});print(json.dumps(hist[-1]),flush=True);write_jsonl(out/'history.jsonl',hist)
 torch.save({k:v.float().cpu() for k,v in model.state_dict().items()},out/'mask_retrieval_head.pt');write_metadata(out/'metadata.json',experiment_metadata(stage='v12_high_fidelity_retrieval_probe',protocol_sha256=sha256(a.protocol_config),selector_sha256=sha256(out/'mask_retrieval_head.pt')))
if __name__=='__main__':main()
