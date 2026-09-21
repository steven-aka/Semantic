#!/usr/bin/env python3
"""V18-L0B-1: 32-query, 32-slot memorization preflight."""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path

def jl(p):
    with open(p) as f:return [json.loads(x) for x in f if x.strip()]

def set_f1(a,b):
    from src.target.answer_parser import parse_answer
    norm=lambda s:{x.strip().lower() for x in parse_answer(s).split('#') if x.strip()}
    x,y=norm(a),norm(b)
    if not x and not y:return 1.0
    if not x or not y:return 0.0
    p=len(x&y)/len(x);r=len(x&y)/len(y)
    return 2*p*r/(p+r) if p+r else 0.0

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=320);ap.add_argument('--lr',type=float,default=1e-3);ap.add_argument('--gpu',default='cuda:0');a=ap.parse_args()
    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.manual_seed(20260921);random.seed(20260921)
    out=Path('results/v2_rank_then_cut/v18_l0b1_tiny_memorization');out.mkdir(parents=True,exist_ok=True)
    root=Path('results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout')
    data={x['example_id']:x for x in jl(root/'data/v10_v12_train_train_clean.jsonl')}
    traces=[x for x in jl('results/v2_rank_then_cut/v17stop_c1_obs0_native_confidence/traces.jsonl') if x['depth']==10]
    teach={x['example_id']:x for x in traces}; ids=sorted(teach)[:32]
    dev=a.gpu
    etok=AutoTokenizer.from_pretrained('models/Qwen3-1.7B',local_files_only=True,trust_remote_code=True)
    enc=AutoModelForCausalLM.from_pretrained('models/Qwen3-1.7B',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,device_map={'':dev}).eval()
    for p in enc.parameters():p.requires_grad_(False)
    cached=[]
    with torch.inference_mode():
      for qid in ids:
        d=data[qid]; text='Question:\n'+d['question']+'\n\nOriginal context:\n'+'\n\n'.join(d['packet_texts'])
        z=etok(text,return_tensors='pt',truncation=True,max_length=2048).to(dev)
        h=enc.model(**z,use_cache=False).last_hidden_state[0].cpu()
        cached.append(h)
    del enc;torch.cuda.empty_cache()
    ttok=AutoTokenizer.from_pretrained('models/Qwen3-8B',local_files_only=True,trust_remote_code=True)
    target=AutoModelForCausalLM.from_pretrained('models/Qwen3-8B',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,device_map={'':dev}).eval()
    for p in target.parameters():p.requires_grad_(False)
    marker='ZZZ_LATENT_MEMORY_ZZZ'
    fixed=[]
    for qid in ids:
      q=data[qid]['question']; user=f'Use only the latent context below to answer the question.\nContext:\n{marker}\n\nQuestion:\n{q}'
      rendered=ttok.apply_chat_template([{'role':'system','content':'Put only the shortest final answer between <answer> and </answer>, without explanation.'},{'role':'user','content':user}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
      pre,suf=rendered.split(marker)
      fixed.append((ttok(pre,add_special_tokens=False,return_tensors='pt').input_ids[0],ttok(suf,add_special_tokens=False,return_tensors='pt').input_ids[0],torch.tensor(teach[qid]['token_ids'],dtype=torch.long)))
    class R(nn.Module):
      def __init__(self):
        super().__init__(); self.inp=nn.Linear(2048,512);self.slots=nn.Parameter(torch.randn(32,512)*.02);self.attn=nn.MultiheadAttention(512,8,batch_first=True);self.ln=nn.LayerNorm(512);self.ff=nn.Sequential(nn.Linear(512,1024),nn.GELU(),nn.Linear(1024,512));self.out=nn.Linear(512,4096)
      def forward(self,h):
        kv=self.inp(h.float()).unsqueeze(0);q=self.slots.unsqueeze(0);x=self.attn(q,kv,kv,need_weights=False)[0];x=self.ln(q+x);return self.out(self.ln(x+self.ff(x)))[0].to(torch.bfloat16)
    r=R().to(dev); opt=torch.optim.AdamW(r.parameters(),lr=a.lr); emb=target.get_input_embeddings()
    def loss_one(i,grad=True):
      pre,suf,ans=fixed[i]; pre,suf,ans=pre.to(dev),suf.to(dev),ans.to(dev); lat=r(cached[i].to(dev)); prompt=torch.cat([emb(pre),lat,emb(suf)],0); allx=torch.cat([prompt,emb(ans)],0).unsqueeze(0)
      logits=target(inputs_embeds=allx,use_cache=False).logits[0]; pred=logits[len(prompt)-1:len(prompt)-1+len(ans)].float()
      return nn.functional.cross_entropy(pred,ans)
    with torch.no_grad(): initial=sum(float(loss_one(i)) for i in range(32))/32
    hist=[];start=time.time()
    for step in range(a.steps):
      i=step%32; opt.zero_grad(set_to_none=True); loss=loss_one(i);loss.backward();torch.nn.utils.clip_grad_norm_(r.parameters(),1.0);opt.step()
      if (step+1)%32==0:
        with torch.no_grad(): cur=sum(float(loss_one(j)) for j in range(32))/32
        hist.append({'step':step+1,'nll':cur});print(hist[-1],flush=True)
    with torch.no_grad(): final=sum(float(loss_one(i)) for i in range(32))/32
    # Greedy cached decoding from the same latent interface.
    rows=[]
    with torch.inference_mode():
      for i,qid in enumerate(ids):
        pre,suf,ans=fixed[i]; lat=r(cached[i].to(dev)); prompt=torch.cat([emb(pre.to(dev)),lat,emb(suf.to(dev))],0).unsqueeze(0)
        o=target(inputs_embeds=prompt,use_cache=True);cache=o.past_key_values; nxt=o.logits[:,-1].argmax(-1); gen=[]
        for _ in range(128):
          t=int(nxt);gen.append(t)
          if t==ttok.eos_token_id:break
          o=target(input_ids=nxt[:,None],past_key_values=cache,use_cache=True);cache=o.past_key_values;nxt=o.logits[:,-1].argmax(-1)
        text=ttok.decode(gen,skip_special_tokens=True); rows.append({'example_id':qid,'generated':text,'teacher':teach[qid]['prediction'],'teacher_f1':set_f1(text,teach[qid]['prediction'])})
    meanf=sum(x['teacher_f1'] for x in rows)/32; reduction=1-final/initial
    norms=r.slots.detach().float().norm(dim=1); rank=int(torch.linalg.matrix_rank(r.slots.detach().float()).item())
    summary={'protocol':'V18-L0B1_TINY_MEMORIZATION','queries':32,'slots':32,'steps':a.steps,'initial_nll':initial,'final_nll':final,'relative_nll_reduction':reduction,'mean_free_generation_teacher_set_f1':meanf,'slot_norm_min':float(norms.min()),'slot_norm_max':float(norms.max()),'slot_effective_rank':rank,'history':hist,'elapsed_seconds':time.time()-start,'target_trainable_parameters':sum(p.requires_grad for p in target.parameters()),'gate':{'nll_reduction_ge_0.70':reduction>=.70,'teacher_f1_ge_0.90':meanf>=.90,'finite':math.isfinite(final),'rank_gt_1':rank>1},'decision':'GO_V18_L0B2_NESTED_MEMORIZATION' if reduction>=.70 and meanf>=.90 and math.isfinite(final) and rank>1 else 'STOP_LATENT_CHANNEL','sealed_sets_read':False}
    (out/'predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');torch.save(r.state_dict(),out/'resampler.pt');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
