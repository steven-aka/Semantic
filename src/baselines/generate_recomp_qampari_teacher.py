"""Generate resumable RECOMP-style QAMPARI summaries with a frozen teacher."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
from src.data.schemas import read_jsonl

def main():
    p=argparse.ArgumentParser();p.add_argument("--inputs",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--model",default="models/Qwen3-14B");p.add_argument("--tensor-parallel-size",type=int,default=2);p.add_argument("--gpu-memory-utilization",type=float,default=.45);p.add_argument("--max-model-len",type=int,default=3072);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--limit",type=int)
    a=p.parse_args();from vllm import LLM,SamplingParams
    rows=list(read_jsonl(a.inputs));rows=rows[:a.limit] if a.limit else rows;a.output.parent.mkdir(parents=True,exist_ok=True)
    done={r["example_id"] for r in read_jsonl(a.output)} if a.output.exists() else set();pending=[r for r in rows if r["example_id"] not in done]
    llm=LLM(model=a.model,tensor_parallel_size=a.tensor_parallel_size,gpu_memory_utilization=a.gpu_memory_utilization,max_model_len=a.max_model_len,trust_remote_code=True)
    tok=llm.get_tokenizer();params=SamplingParams(temperature=0,max_tokens=192)
    with a.output.open("a",encoding="utf-8") as f:
        for i in range(0,len(pending),a.batch_size):
            batch=pending[i:i+a.batch_size];prompts=[]
            for r in batch:
                messages=[{"role":"system","content":"You compress retrieved evidence for a frozen question-answering model. Be concise and strictly source-grounded."},{"role":"user","content":r["teacher_prompt"]}]
                prompts.append(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False))
            t=time.perf_counter();outs=llm.generate(prompts,params,use_tqdm=False);secs=(time.perf_counter()-t)/len(batch)
            for r,o in zip(batch,outs):
                rec={k:r[k] for k in ("example_id","question","context")};rec.update({"summary":o.outputs[0].text.strip(),"teacher_model":a.model,"teacher_seconds":secs})
                f.write(json.dumps(rec,ensure_ascii=False)+"\n")
            f.flush();os.fsync(f.fileno());print(json.dumps({"complete":len(done)+min(i+len(batch),len(pending)),"total":len(rows)}),flush=True)

if __name__=="__main__":main()
