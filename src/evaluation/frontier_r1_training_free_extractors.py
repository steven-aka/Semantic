"""Training-free BM25 and deterministic-random sentence baselines."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, re, time
from collections import Counter
from pathlib import Path
from src.data.schemas import read_jsonl
from src.evaluation.frontier_r1_hard_screen import BASE, LINEAGE, R0, render_selected

OUT=BASE/"frontier_r1_training_free_extractors_screen64"
SENT=re.compile(r"(?<=[.!?])\s+|\n+"); WORD=re.compile(r"[A-Za-z0-9]+")

def rows():
 m=json.loads((R0/"manifest.json").read_text());ids=set(m["screen_query_ids"]);d={r["example_id"]:r for r in read_jsonl(LINEAGE/"data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids};out=[]
 for q in m["screen_query_ids"]:
  packets=d[q]["packet_texts"];units=[x.strip() for p in packets for x in SENT.split(p) if x.strip()]
  out.append({"example_id":q,"question":d[q]["question"],"units":units,"original_context":render_selected(packets,set(range(12)))})
 return out

def bm25_scores(query,docs,k1=1.5,b=.75):
 toks=[[x.lower() for x in WORD.findall(d)] for d in docs];q=[x.lower() for x in WORD.findall(query)];n=len(toks);avg=sum(map(len,toks))/max(1,n);df=Counter(w for doc in toks for w in set(doc));scores=[]
 for doc in toks:
  tf=Counter(doc);s=0.0
  for w in q:
   idf=math.log(1+(n-df[w]+.5)/(df[w]+.5));f=tf[w];s+=idf*(f*(k1+1))/(f+k1*(1-b+b*len(doc)/max(avg,1))) if f else 0
  scores.append(s)
 return scores

def choose(units,order,budget,tok):
 selected=[]
 for i in order:
  trial=sorted(selected+[i]);text="\n\n".join(units[j] for j in trial)
  if len(tok.encode(text,add_special_tokens=False))<=budget:selected.append(i)
 return sorted(selected or [order[0]])

def compress(source):
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained("models/Qwen3-8B",trust_remote_code=True,local_files_only=True);OUT.mkdir(parents=True,exist_ok=True);p=OUT/"compressed.jsonl";done={(r["example_id"],r["arm"],r["keep_rate"]):r for r in read_jsonl(p)} if p.exists() else {}
 with p.open("a",encoding="utf-8") as f:
  for row in source:
   src=len(tok.encode(row["original_context"],add_special_tokens=False));scores=bm25_scores(row["question"],row["units"]);orders={"bm25":sorted(range(len(scores)),key=lambda i:(-scores[i],i))}
   rng=random.Random(int(hashlib.sha256(row["example_id"].encode()).hexdigest()[:16],16));orders["random"]=list(range(len(scores)));rng.shuffle(orders["random"])
   for arm,order in orders.items():
    for rate in (.5,.25,.125):
     key=(row["example_id"],arm,rate)
     if key in done:continue
     ids=choose(row["units"],order,max(1,round(src*rate)),tok);ctx="\n\n".join(row["units"][i] for i in ids);ct=len(tok.encode(ctx,add_special_tokens=False));rec={"example_id":row["example_id"],"question":row["question"],"arm":arm,"keep_rate":rate,"context":ctx,"source_context_tokens":src,"compressed_context_tokens":ct,"actual_keep_rate":ct/src,"selected_unit_indices":ids}
     f.write(json.dumps(rec,ensure_ascii=False)+"\n");f.flush();os.fsync(f.fileno());done[key]=rec
 assert len(done)==384

def evaluate(args,source):
 from src.evaluation.qampari_metrics import parse_list_prediction,qampari_list_metrics
 from src.target.qampari_runner import QampariTargetRunner
 from vllm import SamplingParams
 comp=list(read_jsonl(OUT/"compressed.jsonl"));ids={r["example_id"] for r in source};atoms={r["example_id"]:r["answer_atoms"] for r in read_jsonl(Path("data/units/qampari_rank_v2_candidates5000_annotations.jsonl")) if r["example_id"] in ids};base={r["example_id"]:r for r in read_jsonl(BASE/"frontier_r1_hard_screen64/fresh_original_baseline_outputs.jsonl")};p=OUT/"target_outputs.jsonl";done={(r["example_id"],r["arm"],r["keep_rate"]):r for r in read_jsonl(p)} if p.exists() else {};pending=[r for r in comp if (r["example_id"],r["arm"],r["keep_rate"]) not in done]
 if pending:
  target=QampariTargetRunner("models/Qwen3-8B",backend="vllm",max_new_tokens=256,gpu_memory_utilization=args.gpu_memory_utilization,max_model_len=4096);params=SamplingParams(temperature=0,max_tokens=256)
  with p.open("a",encoding="utf-8") as f:
   for s in range(0,len(pending),64):
    batch=pending[s:s+64];prompts=target._chat_prompts([r["question"] for r in batch],[r["context"] for r in batch]);begin=time.perf_counter();outs=target.model.generate(prompts,params,use_tqdm=False);elapsed=time.perf_counter()-begin
    for x,prompt,o in zip(batch,prompts,outs):
     pred=parse_list_prediction(o.outputs[0].text);rec={k:x[k] for k in ("example_id","arm","keep_rate","source_context_tokens","compressed_context_tokens","actual_keep_rate")};rec.update({"f1":float(qampari_list_metrics(pred,atoms[x["example_id"]])["f1"]),"prediction":pred,"raw_output":o.outputs[0].text,"target_seconds":elapsed/len(batch)});f.write(json.dumps(rec,ensure_ascii=False)+"\n");done[(rec["example_id"],rec["arm"],rec["keep_rate"])]=rec
    f.flush();os.fsync(f.fileno());print(json.dumps({"evaluated":len(done),"total":384}),flush=True)
 results=[]
 for arm in ("bm25","random"):
  for rate in (.125,.25,.5):
   g=[r for (_,a,v),r in done.items() if a==arm and v==rate];b=[base[r["example_id"]]["f1"] for r in g];results.append({"arm":arm,"requested_keep_rate":rate,"mean_actual_keep_rate":sum(r["actual_keep_rate"] for r in g)/64,"mean_context_tokens":sum(r["compressed_context_tokens"] for r in g)/64,"mean_f1":sum(r["f1"] for r in g)/64,"delta_f1":sum(r["f1"]-base[r["example_id"]]["f1"] for r in g)/64,"success_090":sum(r["f1"]+1e-6>=.9 for r in g),"repairs_090":sum(base[r["example_id"]]["f1"]<.9<=r["f1"]+1e-6 for r in g),"breaks_090":sum(base[r["example_id"]]["f1"]+1e-6>=.9>r["f1"]+1e-6 for r in g)})
 summary={"protocol":"FRONTIER-R1-TRAINING-FREE-EXTRACTORS-SCREEN64","baseline_mean_f1":sum(r["f1"] for r in base.values())/64,"baseline_success_090":sum(r["f1"]+1e-6>=.9 for r in base.values()),"results":results,"decision":"SAVE_AS_PAPER_BASELINES_NO_ESCALATION","new_target_calls":len(done),"sealed_sets_read":False};(OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument("--mode",choices=("compress","evaluate","all"),default="all");p.add_argument("--gpu-memory-utilization",type=float,default=.4);a=p.parse_args();source=rows();compress(source) if a.mode in ("compress","all") else None;evaluate(a,source) if a.mode in ("evaluate","all") else None
if __name__=="__main__":main()
