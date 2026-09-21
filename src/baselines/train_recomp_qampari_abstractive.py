"""Fine-tune the RECOMP T5 compressor on frozen teacher summaries; Target is absent."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.data.schemas import read_jsonl

class Summaries(Dataset):
    def __init__(self,path): self.rows=[r for r in read_jsonl(path) if r.get("summary","").strip()]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=Path("configs/baseline_recomp_qampari_abstractive.json"));p.add_argument("--train",type=Path,required=True);p.add_argument("--validation",type=Path,required=True);p.add_argument("--output",type=Path,default=Path("checkpoints/baselines/recomp_qampari_abstractive"));p.add_argument("--device",default="cuda");p.add_argument("--max-steps",type=int)
    a=p.parse_args();c=json.loads(a.config.read_text());random.seed(c["seed"]);torch.manual_seed(c["seed"])
    tok=AutoTokenizer.from_pretrained(c["compressor_initialization"],local_files_only=True)
    model=AutoModelForSeq2SeqLM.from_pretrained(c["compressor_initialization"],local_files_only=True,torch_dtype=torch.bfloat16).to(a.device)
    def collate(rows):
        x=[f"Question: {r['question']}\n Document: {r['context']}\n Summary: " for r in rows]
        enc=tok(x,padding=True,truncation=True,max_length=c["max_source_length"],return_tensors="pt")
        lab=tok(text_target=[r["summary"] for r in rows],padding=True,truncation=True,max_length=c["max_target_length"],return_tensors="pt")["input_ids"]
        lab[lab==tok.pad_token_id]=-100; return {**enc,"labels":lab}
    tr=Summaries(a.train);va=Summaries(a.validation);g=torch.Generator().manual_seed(c["seed"])
    tl=DataLoader(tr,batch_size=c["batch_size"],shuffle=True,generator=g,collate_fn=collate);vl=DataLoader(va,batch_size=c["batch_size"],collate_fn=collate)
    opt=torch.optim.AdamW(model.parameters(),lr=c["learning_rate"],weight_decay=c["weight_decay"]);best=None;hist=[];step=0;a.output.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,c["epochs"]+1):
        model.train();opt.zero_grad(set_to_none=True);losses=[]
        for batch in tl:
            batch={k:v.to(a.device) for k,v in batch.items()};loss=model(**batch).loss/c["gradient_accumulation_steps"];loss.backward();losses.append(float(loss)*c["gradient_accumulation_steps"]);step+=1
            if step%c["gradient_accumulation_steps"]==0: opt.step();opt.zero_grad(set_to_none=True)
            if a.max_steps and step>=a.max_steps: break
        if a.max_steps and step>=a.max_steps: break
        model.eval();vals=[]
        with torch.no_grad():
            for batch in vl:
                batch={k:v.to(a.device) for k,v in batch.items()};vals.append(float(model(**batch).loss))
        rec={"epoch":epoch,"steps":step,"train_loss":sum(losses)/len(losses),"validation_loss":sum(vals)/len(vals)};hist.append(rec);print(json.dumps(rec),flush=True)
        if best is None or rec["validation_loss"]<best[0]:best=(rec["validation_loss"],rec);model.save_pretrained(a.output);tok.save_pretrained(a.output)
        if a.max_steps and step>=a.max_steps: break
    summary={"protocol":c["protocol"],"train_rows":len(tr),"validation_rows":len(va),"history":hist,"selected":best[1] if best else None,"target_loaded":False,"target_trainable":False,"sealed_sets_read":False}
    (a.output/"training_summary.json").write_text(json.dumps(summary,indent=2)+"\n")

if __name__=="__main__":main()
