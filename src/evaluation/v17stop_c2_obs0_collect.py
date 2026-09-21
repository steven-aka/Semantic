"""Collect fixed final-layer H1/H2/H3 states on the 256-query C2 design set."""
from __future__ import annotations
import json, os, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.target.qampari_runner import QAMPARI_SYSTEM_PROMPT, build_qampari_prompt

CFG=Path('configs/v17stop_c2_obs0_internal_preflight.json')
C1=Path('results/v2_rank_then_cut/v17stop_c1_obs0_native_confidence')
DATA=Path('results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/data/v10_v12_train_train_clean.jsonl')
ANN=Path('data/units/qampari_rank_v2_candidates5000_annotations.jsonl')
OUT=Path('results/v2_rank_then_cut/v17stop_c2_obs0_internal_design')

def main():
 cfg=json.loads(CFG.read_text()); cfg['batch_size']=8
 old={(r['example_id'],int(r['depth'])):r for r in read_jsonl(C1/'traces.jsonl')}
 qids=sorted({q for q,_ in old}); ids=set(qids)
 data={r['example_id']:r for r in read_jsonl(DATA) if r['example_id'] in ids}
 atoms={r['example_id']:r['answer_atoms'] for r in read_jsonl(ANN) if r['example_id'] in ids}
 tok=AutoTokenizer.from_pretrained(cfg['target_model'],trust_remote_code=True,local_files_only=True); tok.padding_side='left'
 if tok.pad_token_id is None: tok.pad_token_id=tok.eos_token_id
 model=AutoModelForCausalLM.from_pretrained(cfg['target_model'],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True,device_map=cfg['device']).eval()
 tasks=[]
 for q in qids:
  for d in cfg['depths']:
   s=old[(q,d)]; ctx='\n\n'.join(x.strip() for i,x in enumerate(data[q]['packet_texts']) if s['mask']&(1<<i))
   msgs=[{'role':'system','content':QAMPARI_SYSTEM_PROMPT},{'role':'user','content':build_qampari_prompt(data[q]['question'],ctx)}]
   tasks.append({'example_id':q,'depth':d,'mask':s['mask'],'prompt':tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False),'context_tokens':s['context_tokens']})
 OUT.mkdir(parents=True,exist_ok=True); rows=[]; vectors=[]; elapsed=0.; peak=0
 for start in range(0,len(tasks),cfg['batch_size']):
  batch=tasks[start:start+cfg['batch_size']]; enc=tok([x['prompt'] for x in batch],return_tensors='pt',padding=True).to(cfg['device']); caps=[]
  def hook(_m,_a,o):
   h=o[0] if isinstance(o,tuple) else o; caps.append(h[:,-1,:].detach().to('cpu',dtype=torch.float16))
  handle=model.model.layers[-1].register_forward_hook(hook); torch.cuda.reset_peak_memory_stats(); begin=time.perf_counter()
  with torch.inference_mode(): seq=model.generate(**enc,do_sample=False,max_new_tokens=cfg['max_new_tokens'],pad_token_id=tok.pad_token_id)
  torch.cuda.synchronize(); elapsed+=time.perf_counter()-begin; peak=max(peak,torch.cuda.max_memory_allocated()); handle.remove(); plen=enc['input_ids'].shape[1]
  for i,t in enumerate(batch):
   gen=seq[i,plen:].tolist();
   if tok.eos_token_id in gen: gen=gen[:gen.index(tok.eos_token_id)+1]
   text=tok.decode(gen,skip_special_tokens=True); ans=parse_list_prediction(text); usable=caps[:max(1,len(gen))]; vs=torch.stack([x[i] for x in usable]); pooled=torch.stack([vs[0],vs.mean(0),vs[-1]])
   idx=len(vectors); vectors.append(pooled)
   rows.append({'example_id':t['example_id'],'depth':t['depth'],'mask':t['mask'],'vector_index':idx,'prediction':' # '.join(ans),'answer_count':len(ans),'f1':float(qampari_list_metrics(ans,atoms[t['example_id']])['f1']),'context_tokens':t['context_tokens'],'prompt_tokens':int(enc['attention_mask'][i].sum()),'generated_tokens':len(gen)})
  print(json.dumps({'completed':len(rows),'total':len(tasks)}),flush=True)
 torch.save(torch.stack(vectors),OUT/'hidden_h1_h2_h3.pt')
 with (OUT/'states.jsonl').open('w') as f:
  for r in rows:f.write(json.dumps(r)+'\n')
 summary={'protocol':'V17-STOP-C2-OBS0_INTERNAL_STATE_DESIGN_COLLECTION','queries':len(qids),'states':len(rows),'hidden_shape':list(torch.stack(vectors).shape),'dtype':'float16','elapsed_seconds':elapsed,'peak_vram_bytes':peak,'target_generations':len(rows),'sealed_sets_read':False}
 (OUT/'collection_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
