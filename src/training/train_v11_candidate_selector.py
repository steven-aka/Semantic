from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Dataset
from src.data.schemas import read_jsonl,write_jsonl
from src.model.trajectory_candidate_head import TrajectoryCandidateHead,candidate_set_loss
from src.reproducibility import experiment_metadata,sha256,write_metadata

class Data(Dataset):
 def __init__(self,data,cache):
  self.rows=list(read_jsonl(data));c=torch.load(cache,map_location='cpu',weights_only=True)
  if c['example_ids']!=[r['example_id'] for r in self.rows]:raise ValueError('cache mismatch')
  self.p=c['packets'];self.q=c['questions']
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):return self.rows[i],self.p[i],self.q[i]
def collate(items):
 rows=[x[0] for x in items];m=[];opt=[];idx=[]
 for i,r in enumerate(rows):m+=r['candidate_masks'];opt+=r['optimal_candidates'];idx += [i]*len(r['candidate_masks'])
 return {'rows':rows,'packets':torch.stack([x[1] for x in items]),'questions':torch.stack([x[2] for x in items]),'masks':torch.tensor(m),'optimal':torch.tensor(opt,dtype=torch.bool),'indices':torch.tensor(idx)}
def summary(rows,chosen):
 levels=[.6,.7,.8,.9,.95];per={}
 complete=0;reg=[]
 for li,l in enumerate(levels):
  vals=[]
  for r,c in zip(rows,chosen):
   if li < len(r['outcomes'][c].get('active_levels',[])): pass
  # Active count is recoverable as max contract count among candidates.
  active=[(r,c) for r,c in zip(rows,chosen) if max(x['contracts'] for x in r['outcomes'])>=li+1]
  per[str(l)]={'examples':len(active),'successes':sum(r['outcomes'][c]['contracts']>=li+1 for r,c in active)}
 for r,c in zip(rows,chosen):
  active=max(x['contracts'] for x in r['outcomes']);ok=r['outcomes'][c]['contracts']==active;complete+=ok
  if ok:
   best=min(x['cumulative_tokens'] for x in r['outcomes'] if x['contracts']==active);den=max(1,active*sum([1]))
   # Candidate data stores cumulative tokens but not full tokens; selector gate regret is recomputed by evaluator below.
 return {'per_level':per,'complete_trajectory_successes':complete,'complete_trajectory_success_fraction':complete/len(rows)}
def evaluate(model,data,device,batch):
 loader=DataLoader(data,batch_size=batch,shuffle=False,collate_fn=collate);chosen=[];rows=[]
 model.eval()
 with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
  for b in loader:
   logits=model(b['packets'].to(device),b['questions'].to(device),b['masks'].to(device),b['indices'].to(device));offset=0
   for r in b['rows']:
    n=len(r['candidate_masks']);chosen.append(int(logits[offset:offset+n].argmax()));offset+=n;rows.append(r)
 return rows,chosen

def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol-config',required=True);p.add_argument('--train-data',required=True);p.add_argument('--train-cache',required=True);p.add_argument('--development-data',required=True);p.add_argument('--development-cache',required=True);p.add_argument('--v10-head',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--steps',type=int,default=400);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--seed',type=int,default=20260918);a=p.parse_args()
 protocol=json.loads(Path(a.protocol_config).read_text())
 if protocol.get('status')!='APPROVED_TO_RUN':raise ValueError('protocol is not approved')
 expected={'train_data':protocol['data']['train_candidates'],'train_cache':protocol['data']['train_cache'],'development_data':protocol['data']['development_candidates'],'development_cache':protocol['data']['development_cache'],'v10_head':protocol['frozen']['v10_head'],'output_dir':protocol['output'],'steps':protocol['optimization']['steps'],'batch_size':protocol['optimization']['batch_size'],'lr':protocol['optimization']['learning_rate'],'seed':protocol['optimization']['seed']}
 mismatch={k:(getattr(a,k),v) for k,v in expected.items() if getattr(a,k)!=v}
 if mismatch:raise ValueError(f'arguments differ from protocol: {mismatch}')
 random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);device=torch.device('cuda:0');out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);write_metadata(out/'protocol_snapshot.json',protocol)
 train=Data(a.train_data,a.train_cache);dev=Data(a.development_data,a.development_cache);model=TrajectoryCandidateHead(model_dim=512,layers=2,heads=8,dropout=.1)
 old=torch.load(a.v10_head,map_location='cpu',weights_only=True);state=model.state_dict();
 for k in state:
  if k in old and state[k].shape==old[k].shape:state[k]=old[k]
 model.load_state_dict(state);model.to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps);loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed),collate_fn=collate);it=iter(loader);hist=[]
 for step in range(1,a.steps+1):
  model.train()
  try:b=next(it)
  except StopIteration:it=iter(loader);b=next(it)
  with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=model(b['packets'].to(device),b['questions'].to(device),b['masks'].to(device),b['indices'].to(device));loss=candidate_set_loss(logits,b['optimal'].to(device),b['indices'].to(device))
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();sched.step();opt.zero_grad(set_to_none=True)
  if step%20==0:hist.append({'step':step,'candidate_set_loss':float(loss)});print(json.dumps(hist[-1]),flush=True);write_jsonl(out/'history.jsonl',hist)
 torch.save({k:v.float().cpu() for k,v in model.state_dict().items()},out/'candidate_selector.pt')
 rows,chosen=evaluate(model,dev,device,a.batch_size);details=[]
 for r,c in zip(rows,chosen):details.append({'example_id':r['example_id'],'selected_index':c,'selected_mask':r['candidate_masks'][c],'outcome':r['outcomes'][c],'optimal_selected':r['optimal_candidates'][c]})
 write_jsonl(out/'development_selection.jsonl',details);write_metadata(out/'selection_summary.json',{'examples':len(details),'optimal_candidate_selected':sum(x['optimal_selected'] for x in details),'outcome_contract_histogram':{str(k):sum(x['outcome']['contracts']==k for x in details) for k in range(6)}})
 write_metadata(out/'metadata.json',experiment_metadata(stage='v11_candidate_selector',protocol_sha256=sha256(a.protocol_config),seed=a.seed,steps=a.steps,train_data_sha256=sha256(a.train_data),development_data_sha256=sha256(a.development_data),v10_head_sha256=sha256(a.v10_head),selector_sha256=sha256(out/'candidate_selector.pt')))
if __name__=='__main__':main()
