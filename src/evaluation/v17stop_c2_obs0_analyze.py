"""Final query-grouped linear-probe gate for C2 internal observability."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch
from src.data.schemas import read_jsonl

ROOT=Path('results/v2_rank_then_cut/v17stop_c2_obs0_internal_design'); LEVELS=(.6,.7,.8,.9,.95); ED={.6:6,.7:7,.8:7,.9:9,.95:10}; TH=(.5,.7,.8,.9,.95,.98,.99)
def fold(q):return int(hashlib.sha256(('C2-OBS0|'+q).encode()).hexdigest(),16)%4
def answers(x):return {z.strip().lower() for z in x.split('#') if z.strip()}
def fit(train,test,key):
 torch.set_num_threads(1); x=torch.stack([e[key] for e in train]).float(); y=torch.tensor([e['y'] for e in train]).float(); xt=torch.stack([e[key] for e in test]).float(); mu=x.mean(0); sd=x.std(0); sd[sd<1e-5]=1; x=(x-mu)/sd;xt=(xt-mu)/sd;w=torch.zeros(x.shape[1],requires_grad=True);b=torch.zeros((),requires_grad=True);o=torch.optim.Adam([w,b],lr=.03)
 for _ in range(400):
  loss=torch.nn.functional.binary_cross_entropy_with_logits(x@w+b,y)+1e-4*w.square().sum();o.zero_grad();loss.backward();o.step()
 with torch.no_grad():return torch.sigmoid(xt@w+b).tolist()
def main():
 rows=list(read_jsonl(ROOT/'states.jsonl')); by={(r['example_id'],r['depth']):r for r in rows}; hs=torch.load(ROOT/'hidden_h1_h2_h3.pt',map_location='cpu'); qids=sorted({r['example_id'] for r in rows}); ex=[]
 for q in qids:
  for tau in LEVELS[:-1]:
   d=ED[tau];r=by[(q,d)];prev=by.get((q,{6:0,7:6,9:7}[d]));a=answers(r['prediction']);p=answers(prev['prediction']) if prev else set();u=a|p;j=len(a&p)/len(u) if u else 1.; out=torch.tensor([tau,d/10,r['answer_count'],r['generated_tokens'],len(a),j,len(a-p),len(p-a)],dtype=torch.float32);v=hs[r['vector_index']]
   ex.append({'q':q,'tau':tau,'y':r['f1']+1e-12>=tau,'out':out,'h1':torch.cat([torch.tensor([tau,d/10]),v[0]]),'h2':torch.cat([torch.tensor([tau,d/10]),v[1]]),'h3':torch.cat([torch.tensor([tau,d/10]),v[2]]),'out_h2':torch.cat([out,v[1]])})
 keys=('out','h1','h2','h3','out_h2');scores={k:{} for k in keys}
 for f in range(4):
  tr=[e for e in ex if fold(e['q'])!=f];te=[e for e in ex if fold(e['q'])==f]
  for k in keys:
   for e,p in zip(te,fit(tr,te,k)):scores[k][(e['q'],e['tau'])]=p
 def fixed(ds):
  s={str(t):0 for t in LEVELS};comp=ctx=cost=0
  for q in qids:
   ok=[]
   for t,d in zip(LEVELS,ds):r=by[(q,d)];z=r['f1']+1e-12>=t;s[str(t)]+=z;ok.append(z);ctx+=r['context_tokens'];cost+=r['prompt_tokens']+r['generated_tokens']
   comp+=all(ok)
  return {'success':s,'complete':comp,'context':ctx/len(qids),'compute':cost/len(qids)}
 d10=fixed([10]*5);ag=fixed([6,7,7,9,10]);oracle_ctx=oracle_cost=0
 for q in qids:
  for t in LEVELS:
   e=by[(q,ED[t])];l=by[(q,10)];stop=e['f1']+1e-12>=t;oracle_ctx+=(e if stop else l)['context_tokens'];oracle_cost+=e['prompt_tokens']+e['generated_tokens'];
   if not stop and ED[t]!=10:oracle_cost+=l['prompt_tokens']+l['generated_tokens']
 oracle={'context':oracle_ctx/len(qids),'compute':oracle_cost/len(qids)}; curves={}
 for k in keys:
  curves[k]=[]
  for th in TH:
   suc={str(t):0 for t in LEVELS};co=ctx=cost=calls=0
   for q in qids:
    oks=[]
    for t in LEVELS:
     e=by[(q,ED[t])];l=by[(q,10)];stop=True if t==.95 else scores[k][(q,t)]>=th;r=e if stop else l;z=r['f1']+1e-12>=t;suc[str(t)]+=z;oks.append(z);ctx+=r['context_tokens'];cost+=e['prompt_tokens']+e['generated_tokens'];calls+=1
     if not stop:cost+=l['prompt_tokens']+l['generated_tokens'];calls+=1
    co+=all(oks)
   ctx/=len(qids);cost/=len(qids); quality=all(suc[str(t)]>=d10['success'][str(t)]-2.56 for t in LEVELS) and co>=d10['complete']-2.56; gap=(d10['compute']-oracle['compute']); capture=(d10['compute']-cost)/gap if gap else -99; go=quality and cost<=.98*d10['compute'] and capture>=.85
   curves[k].append({'threshold':th,'success':suc,'complete':co,'context':ctx,'compute':cost,'calls_per_request':calls/(len(qids)*5),'quality_pass':quality,'compute_oracle_capture':capture,'go':go})
 decision='GO_C2_INTERNAL_VALIDATION' if any(x['go'] for k in keys for x in curves[k]) else 'STOP_INTERNAL_OBSERVABILITY'
 out={'protocol':'V17-STOP-C2-OBS0_INTERNAL_STATE_SUFFICIENCY_CEILING','queries':len(qids),'baselines':{'aggressive':ag,'depth10':d10,'oracle':oracle},'curves':curves,'decision':decision,'gate':'quality within 1pp; compute <=98% depth10; >=85% compute-oracle capture','native_same_path_missing':True,'sealed_sets_read':False};(ROOT/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'decision':decision,'baselines':out['baselines'],'best':{k:max(v,key=lambda x:x['compute_oracle_capture'] if x['quality_pass'] else -999) for k,v in curves.items()}},indent=2))
if __name__=='__main__':main()
