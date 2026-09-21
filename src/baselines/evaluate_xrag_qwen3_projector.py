"""Screen-64 evaluation for the xRAG-style frozen-Qwen projector port."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import torch
from transformers import AutoModel,AutoModelForCausalLM,AutoTokenizer
from src.baselines.frontier_r1_hard_screen import BASE,LINEAGE,R0,render_selected
from src.baselines.train_xrag_qwen3_projector import Projector,pool
from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction,qampari_list_metrics
from src.target.qampari_runner import QAMPARI_SYSTEM_PROMPT

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=Path("configs/baseline_xrag_qwen3_projector.json"));p.add_argument("--checkpoint",type=Path,default=Path("checkpoints/baselines/xrag_qwen3_projector/projector.pt"));p.add_argument("--output",type=Path,default=Path("results/v2_rank_then_cut/baseline_xrag_qwen3_projector/screen64"));p.add_argument("--device",default="cuda");a=p.parse_args();c=json.loads(a.config.read_text());device=torch.device(a.device);a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((R0/"manifest.json").read_text());ids=manifest["screen_query_ids"];wanted=set(ids);data={r["example_id"]:r for r in read_jsonl(LINEAGE/"data/v10_v12_train_train_clean.jsonl") if r["example_id"] in wanted};atoms={r["example_id"]:r["answer_atoms"] for r in read_jsonl(Path("data/units/qampari_rank_v2_candidates5000_annotations.jsonl")) if r["example_id"] in wanted};baseline={r["example_id"]:r for r in read_jsonl(BASE/"frontier_r1_hard_screen64/fresh_original_baseline_outputs.jsonl")}
    rt=AutoTokenizer.from_pretrained(c["retriever"],local_files_only=True);retr=AutoModel.from_pretrained(c["retriever"],local_files_only=True).to(device).eval();qt=AutoTokenizer.from_pretrained(c["target"],trust_remote_code=True,local_files_only=True);target=AutoModelForCausalLM.from_pretrained(c["target"],trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16).to(device).eval();proj=Projector(retr.config.hidden_size,target.config.hidden_size).to(device=device,dtype=torch.bfloat16);proj.load_state_dict(torch.load(a.checkpoint,map_location=device,weights_only=True)["projector"]);proj.eval()
    marker="<XRAG_CONTEXT_VECTOR>";out=a.output/"target_outputs.jsonl";done={r["example_id"]:r for r in read_jsonl(out)} if out.exists() else {}
    with out.open("a") as f,torch.no_grad():
        for q in ids:
            if q in done:continue
            r=data[q];context="\n\n".join(r["packet_texts"]);rb=rt(context,truncation=True,max_length=c["retriever_max_length"],return_tensors="pt").to(device);rv=pool(retr(**rb).last_hidden_state,rb["attention_mask"]);rendered=qt.apply_chat_template([{"role":"system","content":QAMPARI_SYSTEM_PROMPT},{"role":"user","content":f"Compressed context: {marker}\nQuestion: {r['question']}"}],tokenize=False,add_generation_prompt=True,enable_thinking=False);left,right=rendered.split(marker);li=qt.encode(left,add_special_tokens=False);ri=qt.encode(right,add_special_tokens=False);ii=torch.tensor([li+[qt.pad_token_id or qt.eos_token_id]+ri],device=device);ee=target.get_input_embeddings()(ii);ee[:,len(li),:]=proj(rv.to(proj.net[0].weight.dtype)).to(ee.dtype);t=time.perf_counter();gen=target.generate(inputs_embeds=ee,attention_mask=torch.ones_like(ii),max_new_tokens=256,do_sample=False,pad_token_id=qt.eos_token_id);secs=time.perf_counter()-t;text=qt.decode(gen[0],skip_special_tokens=True);pred=parse_list_prediction(text);rec={"example_id":q,"raw_output":text,"prediction":pred,"f1":float(qampari_list_metrics(pred,atoms[q])["f1"]),"latent_context_tokens":1,"target_seconds":secs};f.write(json.dumps(rec,ensure_ascii=False)+"\n");f.flush();done[q]=rec;print(json.dumps({"complete":len(done),"total":64}),flush=True)
    rows=[done[q] for q in ids];thresholds=(.6,.7,.8,.9,.95);summary={"protocol":c["protocol"],"qualification":c["qualification"],"target":"frozen Qwen3-8B","queries":64,"mean_f1":sum(r["f1"] for r in rows)/64,"mean_baseline_f1":sum(baseline[q]["f1"] for q in ids)/64,"latent_context_tokens":1,"threshold_success":{str(t):sum(r["f1"]+1e-6>=t for r in rows) for t in thresholds},"baseline_threshold_success":{str(t):sum(baseline[q]["f1"]+1e-6>=t for q in ids) for t in thresholds},"sealed_sets_read":False};(a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
