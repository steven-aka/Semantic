from __future__ import annotations

import argparse,json,random,time
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from src.data.schemas import read_jsonl,write_jsonl
from src.reproducibility import experiment_metadata,sha256,write_metadata


def render(question,texts,pair):
    def selected(mask):return [f"Packet {i}: {texts[i]}" for i in range(12) if mask&(1<<i)]
    retained=pair["candidate_mask"]&pair["matched_fallback_mask"]
    # Put the causal intervention before the long retained context so truncation
    # can never hide the added/removed evidence from the positive control.
    return "\n\n".join([f"Question: {question}","Added evidence:\n"+"\n".join(selected(pair["added_mask"])),"Removed evidence:\n"+"\n".join(selected(pair["removed_mask"])),"Predict how this evidence swap changes target-model answer fidelity.","Retained evidence:\n"+"\n".join(selected(retained))])


class Head(nn.Module):
    def __init__(self,hidden):super().__init__();self.net=nn.Sequential(nn.Linear(hidden,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,19))
    def forward(self,x):
        y=self.net(x);return y[:,:4],y[:,4:].reshape(-1,5,3)


def load(args):
    from peft import PeftModel,prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True);tok.pad_token=tok.eos_token;tok.padding_side="right"
    base=AutoModelForCausalLM.from_pretrained(args.model,trust_remote_code=True,torch_dtype=torch.bfloat16,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True),device_map={"":0})
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True);lm=PeftModel.from_pretrained(base,Path(args.checkpoint)/"adapter",is_trainable=True);lm.config.use_cache=False
    for n,p in lm.named_parameters():p.requires_grad="lora_" in n
    return tok,lm


def flatten(pair_path,text_path):
    source={r["example_id"]:r for r in read_jsonl(text_path)};items=[]
    for row in read_jsonl(pair_path):
        s=source[row["example_id"]];base=row["fallback"]
        for pair in row["pairs"]:
            items.append({"example_id":row["example_id"],"prompt":render(s["question"],s["packet_texts"],pair),"class":pair["class_index"],"anchor":[int(pair["candidate_success"][i])-int(base["success"][i])+1 for i in range(5)],"active":base["active"],"class_name":pair["class_name"]})
    return items


def batches(items,size,seed,steps):
    rng=random.Random(seed);by={c:[x for x in items if x["class"]==c] for c in range(4)}
    weights=[.35,.25,.25,.15]
    for _ in range(steps):
        yield [rng.choice(by[rng.choices(range(4),weights)[0]]) for _ in range(size)]


def encode(tok,items,max_length,device):
    value=tok([x["prompt"] for x in items],padding=True,truncation=True,max_length=max_length,return_tensors="pt");return {k:v.to(device) for k,v in value.items()}


def forward(lm,head,tokens):
    out=lm(**tokens,output_hidden_states=True,use_cache=False);last=out.hidden_states[-1];index=tokens["attention_mask"].sum(1)-1;pooled=last[torch.arange(len(index),device=last.device),index];return head(pooled)


def loss_fn(c,a,items,device):
    ct=torch.tensor([x["class"] for x in items],device=device);at=torch.tensor([x["anchor"] for x in items],device=device);active=torch.tensor([x["active"] for x in items],device=device)
    ce=torch.nn.functional.cross_entropy(c.float(),ct);ae=torch.nn.functional.cross_entropy(a.float().reshape(-1,3),at.reshape(-1),reduction="none").reshape(-1,5);return ce+.5*(ae*active).sum()/active.sum().clamp_min(1)


def evaluate(tok,lm,head,items,args):
    result=[];lm.eval();head.eval()
    with torch.no_grad():
        for start in range(0,len(items),args.eval_batch_size):
            batch=items[start:start+args.eval_batch_size];tokens=encode(tok,batch,args.max_length,torch.device("cuda:0"))
            with torch.autocast("cuda",dtype=torch.bfloat16):c,a=forward(lm,head,tokens)
            cp=torch.softmax(c.float(),1).cpu();ap=torch.softmax(a.float(),2).cpu()
            result.extend({"example_id":x["example_id"],"class_name":x["class_name"],"class_index":x["class"],"probabilities":p.tolist(),"anchor_delta_probabilities":q.tolist()} for x,p,q in zip(batch,cp,ap))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--protocol-config",required=True);p.add_argument("--train-pairs",required=True);p.add_argument("--train-text",required=True);p.add_argument("--validation-pairs",required=True);p.add_argument("--validation-text",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--model",default="models/Qwen3-4B");p.add_argument("--steps",type=int,default=300);p.add_argument("--batch-size",type=int,default=4);p.add_argument("--gradient-accumulation",type=int,default=4);p.add_argument("--eval-batch-size",type=int,default=8);p.add_argument("--max-length",type=int,default=1024);p.add_argument("--lora-lr",type=float,default=1e-5);p.add_argument("--head-lr",type=float,default=1e-4);p.add_argument("--seed",type=int,default=20260918);a=p.parse_args()
    protocol=json.loads(Path(a.protocol_config).read_text());expected=protocol["arguments"]
    if protocol["status"]!="APPROVED_TO_RUN":raise ValueError("protocol not approved")
    mismatch={k:(getattr(a,k),v) for k,v in expected.items() if getattr(a,k)!=v}
    if mismatch:raise ValueError(mismatch)
    random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);device=torch.device("cuda:0");out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);write_metadata(out/"protocol_snapshot.json",protocol)
    train=flatten(a.train_pairs,a.train_text);valid=flatten(a.validation_pairs,a.validation_text);tok,lm=load(a);head=Head(lm.config.hidden_size).to(device,dtype=torch.bfloat16);lora=[x for x in lm.parameters() if x.requires_grad];optimizer=torch.optim.AdamW([{"params":lora,"lr":a.lora_lr},{"params":head.parameters(),"lr":a.head_lr}],weight_decay=.01)
    from transformers import get_cosine_schedule_with_warmup
    scheduler=get_cosine_schedule_with_warmup(optimizer,max(10,a.steps//10),a.steps);iterator=batches(train,a.batch_size,a.seed,a.steps*a.gradient_accumulation);history=[];started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
    for step in range(1,a.steps+1):
        lm.train();head.train();total=0
        for _ in range(a.gradient_accumulation):
            batch=next(iterator);tokens=encode(tok,batch,a.max_length,device)
            with torch.autocast("cuda",dtype=torch.bfloat16):c,anchor=forward(lm,head,tokens);loss=loss_fn(c,anchor,batch,device)/a.gradient_accumulation
            loss.backward();total+=float(loss.detach())
        torch.nn.utils.clip_grad_norm_([*lora,*head.parameters()],1);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
        if step%25==0:record={"step":step,"loss":total};history.append(record);write_jsonl(out/"history.jsonl",history);print(json.dumps(record),flush=True)
    predictions=evaluate(tok,lm,head,valid,a);write_jsonl(out/"internal_validation_predictions.jsonl",predictions);break_max=max(x["probabilities"][1] for x in predictions if x["class_name"]=="break");repairs=[x for x in predictions if x["class_name"]=="repair"];recovered=sum(x["probabilities"][1]>break_max for x in repairs)
    summary={"complete":True,"scientific_role":"raw-text LoRA positive control on fixed internal validation; diagnostic only","repair_recall_at_zero_pair_break":recovered/len(repairs),"repairs_recovered":recovered,"repairs":len(repairs),"max_break_repair_probability":break_max,"repair_probabilities":[x["probabilities"][1] for x in repairs],"elapsed_seconds":time.perf_counter()-started};write_metadata(out/"summary.json",summary);lm.save_pretrained(out/"adapter");torch.save({k:v.float().cpu() for k,v in head.state_dict().items()},out/"head.pt");write_metadata(out/"metadata.json",experiment_metadata(stage="v16a_raw_text_positive_control",locked_roles_used=False,development300_used=False));print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
