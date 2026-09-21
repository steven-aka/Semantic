"""xRAG-style one-token projector for a strictly frozen Qwen3 Target."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from src.data.schemas import read_jsonl
from src.target.qampari_runner import QAMPARI_SYSTEM_PROMPT

def pool(h,m):
    m=m.unsqueeze(-1).to(h.dtype);return (h*m).sum(1)/m.sum(1).clamp_min(1)

class Projector(nn.Module):
    def __init__(self,di,do):super().__init__();self.net=nn.Sequential(nn.Linear(di,do),nn.GELU(),nn.Linear(do,do))
    def forward(self,x):return self.net(x)

class Rows(Dataset):
    def __init__(self,path,anns):self.rows=list(read_jsonl(path));self.anns=anns
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];return r,[x["answer_text"] for x in self.anns[r["example_id"]]["answer_atoms"]]

def fingerprint(model):
    """Cheap mutation check; optimizer exclusion and zero gradients are the hard guarantees."""
    with torch.no_grad():
        return [float(p.detach().float().reshape(-1)[:16].sum()) for p in list(model.parameters())[::64]]

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=Path("configs/baseline_xrag_qwen3_projector.json"));p.add_argument("--data",type=Path);p.add_argument("--output",type=Path,default=Path("checkpoints/baselines/xrag_qwen3_projector"));p.add_argument("--device",default="cuda");p.add_argument("--max-steps",type=int);p.add_argument("--preflight",action="store_true")
    a=p.parse_args();c=json.loads(a.config.read_text());random.seed(c["seed"]);torch.manual_seed(c["seed"]);device=torch.device(a.device)
    rt=AutoTokenizer.from_pretrained(c["retriever"],local_files_only=True);retr=AutoModel.from_pretrained(c["retriever"],local_files_only=True).to(device).eval()
    qt=AutoTokenizer.from_pretrained(c["target"],trust_remote_code=True,local_files_only=True);target=AutoModelForCausalLM.from_pretrained(c["target"],trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16).to(device).eval()
    for model in (retr,target):
        for x in model.parameters():x.requires_grad_(False)
    proj=Projector(retr.config.hidden_size,target.config.hidden_size).to(device=device,dtype=torch.bfloat16)
    before=fingerprint(target);trainable=[n for n,pv in proj.named_parameters() if pv.requires_grad]
    anns={x["example_id"]:x for x in read_jsonl(Path("data/units/qampari_rank_v2_candidates5000_annotations.jsonl"))}
    source=a.data or Path(c["train_source"]);row=next(iter(read_jsonl(source)));context="\n\n".join(row["packet_texts"])
    answers=[x["answer_text"] for x in anns[row["example_id"]]["answer_atoms"]]
    rb=rt(context,truncation=True,max_length=c["retriever_max_length"],return_tensors="pt").to(device)
    with torch.no_grad():rv=pool(retr(**rb).last_hidden_state,rb["attention_mask"])
    marker="<XRAG_CONTEXT_VECTOR>"
    rendered=qt.apply_chat_template([{"role":"system","content":QAMPARI_SYSTEM_PROMPT},{"role":"user","content":f"Compressed context: {marker}\nQuestion: {row['question']}"}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    left,right=rendered.split(marker)
    left_ids=qt.encode(left,add_special_tokens=False);right_ids=qt.encode(right,add_special_tokens=False)
    prefix=left_ids+[qt.pad_token_id or qt.eos_token_id]+right_ids;latent_position=len(left_ids)
    answer=qt.encode("<answer>"+"#".join(answers)+"</answer>"+qt.eos_token,add_special_tokens=False)[:c["answer_max_length"]]
    ids=torch.tensor([prefix+answer],device=device);labels=torch.full_like(ids,-100);labels[:,len(prefix):]=ids[:,len(prefix):]
    emb=target.get_input_embeddings()(ids).detach();emb[:,latent_position,:]=proj(rv.to(proj.net[0].weight.dtype)).to(emb.dtype)
    out=target(inputs_embeds=emb,attention_mask=torch.ones_like(ids),labels=labels,use_cache=False);out.loss.backward()
    grad_ok=all(x.grad is not None and torch.isfinite(x.grad).all() for x in proj.parameters());target_grad=sum(x.grad is not None for x in target.parameters());after=fingerprint(target)
    result={"protocol":c["protocol"],"mode":"preflight" if a.preflight else "train","loss":float(out.loss),"projector_trainable_parameters":sum(x.numel() for x in proj.parameters()),"trainable_names":trainable,"projector_gradients_finite":grad_ok,"target_parameter_grads":target_grad,"target_hash_unchanged":before==after,"retriever_trainable":False,"latent_context_tokens":1,"peak_cuda_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None}
    a.output.mkdir(parents=True,exist_ok=True);(a.output/"preflight.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
    if a.preflight:return
    # Full training deliberately follows only after the mutation/gradient preflight above.
    proj.zero_grad(set_to_none=True);opt=torch.optim.AdamW(proj.parameters(),lr=c["learning_rate"],weight_decay=c["weight_decay"])
    rows=Rows(source,anns);validation=Rows(Path(c["validation_source"]),anns);order=list(range(len(rows)));rng=random.Random(c["seed"]);step=0;losses=[];history=[];best=None
    def compute(example, use_grad):
        r,ans=example;ctx="\n\n".join(r["packet_texts"]);rb=rt(ctx,truncation=True,max_length=c["retriever_max_length"],return_tensors="pt").to(device)
        with torch.no_grad():rv=pool(retr(**rb).last_hidden_state,rb["attention_mask"])
        rendered=qt.apply_chat_template([{"role":"system","content":QAMPARI_SYSTEM_PROMPT},{"role":"user","content":f"Compressed context: {marker}\nQuestion: {r['question']}"}],tokenize=False,add_generation_prompt=True,enable_thinking=False);left,right=rendered.split(marker);li=qt.encode(left,add_special_tokens=False);ri=qt.encode(right,add_special_tokens=False);pre=li+[qt.pad_token_id or qt.eos_token_id]+ri;pos=len(li);answer_ids=qt.encode("<answer>"+"#".join(ans)+"</answer>"+qt.eos_token,add_special_tokens=False)[:c["answer_max_length"]];ii=torch.tensor([pre+answer_ids],device=device);lab=torch.full_like(ii,-100);lab[:,len(pre):]=ii[:,len(pre):];ee=target.get_input_embeddings()(ii).detach();ee[:,pos,:]=proj(rv.to(proj.net[0].weight.dtype)).to(ee.dtype)
        return target(inputs_embeds=ee,attention_mask=torch.ones_like(ii),labels=lab,use_cache=False).loss
    for epoch in range(c["epochs"]):
        rng.shuffle(order)
        for ix in order:
            loss=compute(rows[ix],True)/c["gradient_accumulation_steps"];loss.backward();losses.append(float(loss.detach())*c["gradient_accumulation_steps"]);step+=1
            if step%c["gradient_accumulation_steps"]==0:opt.step();opt.zero_grad(set_to_none=True)
            if step%50==0:print(json.dumps({"epoch":epoch+1,"step":step,"mean_loss_50":sum(losses[-50:])/min(50,len(losses))}),flush=True)
            if a.max_steps and step>=a.max_steps:break
        if a.max_steps and step>=a.max_steps:break
        proj.eval();vals=[]
        with torch.no_grad():
            for ix in range(len(validation)):vals.append(float(compute(validation[ix],False)))
        proj.train();rec={"epoch":epoch+1,"steps":step,"train_loss":sum(losses[-len(rows):])/min(len(rows),len(losses)),"validation_nll":sum(vals)/len(vals)};history.append(rec);print(json.dumps(rec),flush=True)
        if best is None or rec["validation_nll"]<best[0]:best=(rec["validation_nll"],rec);torch.save({"projector":proj.state_dict(),"config":c,"steps":step},a.output/"projector.pt")
    if best is None:torch.save({"projector":proj.state_dict(),"config":c,"steps":step},a.output/"projector.pt")
    final={**result,"mode":"train","steps":step,"mean_train_loss":sum(losses)/len(losses),"history":history,"selected":best[1] if best else None,"target_hash_unchanged_after_training":before==fingerprint(target)};(a.output/"training_summary.json").write_text(json.dumps(final,indent=2)+"\n");print(json.dumps(final,indent=2))

if __name__=="__main__":main()
