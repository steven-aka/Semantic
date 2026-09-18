from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from src.data.schemas import ExactSearchResult,read_jsonl,write_jsonl
from src.search.atomic_nested_chain import state_to_mask
from src.model.mask_value_head import MaskValueHead
from src.reproducibility import sha256,write_metadata

def project(order,mask): return [p for p in order if mask>>p&1]+[p for p in order if not mask>>p&1]
def outcome(order,attained,tokens,active):
 masks=[0]
 for p in order:masks.append(masks[-1]|1<<p)
 selected=[]
 for level in range(1,active+1):
  feasible=[tokens[m] for m in masks if attained[m]>=level]
  selected.append(min(feasible) if feasible else None)
 return sum(x is not None for x in selected),sum(x for x in selected if x is not None)
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--cache',required=True);p.add_argument('--rollouts',required=True);p.add_argument('--exact-dir',required=True);p.add_argument('--v10-head',required=True);p.add_argument('--output',required=True);p.add_argument('--manifest',required=True);p.add_argument('--top-k',type=int,default=16);p.add_argument('--batch-size',type=int,default=16);a=p.parse_args()
 rows=list(read_jsonl(a.data)); cache=torch.load(a.cache,map_location='cpu',weights_only=True);orders={r['example_id']:r['decoded_order'] for r in read_jsonl(a.rollouts)}
 if cache['example_ids']!=[r['example_id'] for r in rows] or set(orders)!={r['example_id'] for r in rows}:raise ValueError('ID mismatch')
 device=torch.device('cuda:0');h=MaskValueHead(model_dim=512,layers=2,heads=8,dropout=.1);h.load_state_dict(torch.load(a.v10_head,map_location='cpu',weights_only=True));h.to(device).eval();out=[]
 with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
  for start in range(0,len(rows),a.batch_size):
   stop=min(start+a.batch_size,len(rows));pack=cache['packets'][start:stop].to(device);q=cache['questions'][start:stop].to(device);n=stop-start;score=torch.empty((n,4096),device='cpu')
   for ms in range(0,4096,512):
    width=min(512,4096-ms);m=torch.arange(ms,ms+width,device=device).repeat(n);idx=torch.arange(n,device=device).repeat_interleave(width);score[:,ms:ms+width]=h(pack,q,m,idx)[:,3].reshape(n,width).float().cpu()
   for j,row in enumerate(rows[start:stop]):
    candidates=[0]+torch.argsort(score[j],descending=True,stable=True)[:a.top_k].tolist();candidates=list(dict.fromkeys(candidates));base=orders[row['example_id']]
    exact=list(read_jsonl(Path(a.exact_dir)/(row['example_id']+'.jsonl'),ExactSearchResult));by={state_to_mask(x.state):x for x in exact};levels=[float(x) for x in row['attainable_levels']];att=[sum(by[m].fidelity+1e-12>=level for level in levels) for m in range(4096)];tok=[int(by[m].tokens) for m in range(4096)]
    vals=[outcome(project(base,m),att,tok,len(levels)) for m in candidates];best=min(vals,key=lambda x:(-x[0],x[1]));out.append({'example_id':row['example_id'],'candidate_masks':candidates,'outcomes':[{'contracts':x[0],'cumulative_tokens':x[1]} for x in vals],'optimal_candidates':[x==best for x in vals]})
   print(json.dumps({'built':stop,'total':len(rows)}),flush=True)
 write_jsonl(a.output,out);write_metadata(a.manifest,{'complete':True,'examples':len(out),'top_k':a.top_k,'data':a.data,'data_sha256':sha256(a.data),'cache_sha256':sha256(a.cache),'rollouts_sha256':sha256(a.rollouts),'v10_head_sha256':sha256(a.v10_head),'output_sha256':sha256(a.output)})
if __name__=='__main__':main()
