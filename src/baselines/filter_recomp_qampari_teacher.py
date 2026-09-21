"""Apply the frozen RECOMP distillation filter and report coverage."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
from transformers import AutoTokenizer
from src.data.schemas import read_jsonl

def norm(x):return re.sub(r"[^a-z0-9]+"," ",x.lower()).strip()
def main():
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--minimum-answer-recall",type=float,default=.6);p.add_argument("--max-target-tokens",type=int,default=256);a=p.parse_args()
    anns={r["example_id"]:r for r in read_jsonl(Path("data/units/qampari_rank_v2_candidates5000_annotations.jsonl"))};tok=AutoTokenizer.from_pretrained("models/recomp-nq-abstractive",local_files_only=True);kept=[];stats=[]
    for r in read_jsonl(a.input):
        s=norm(r["summary"]);atoms=anns[r["example_id"]]["answer_atoms"];covered=sum(any(norm(x) in s for x in [z["answer_text"],*z.get("aliases",[])]) for z in atoms);recall=covered/max(1,len(atoms));n=len(tok(r["summary"],add_special_tokens=True).input_ids);ok=bool(r["summary"].strip()) and recall>=a.minimum_answer_recall and n<=a.max_target_tokens
        q={**r,"teacher_answer_recall":recall,"summary_tokens":n,"filter_pass":ok};stats.append(q);kept.append(q) if ok else None
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w") as f:
        for r in kept:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    report={"input_rows":len(stats),"kept_rows":len(kept),"keep_fraction":len(kept)/max(1,len(stats)),"minimum_answer_recall":a.minimum_answer_recall,"max_target_tokens":a.max_target_tokens,"mean_answer_recall":sum(x["teacher_answer_recall"] for x in stats)/max(1,len(stats)),"mean_summary_tokens":sum(x["summary_tokens"] for x in stats)/max(1,len(stats))}
    a.output.with_suffix(".summary.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
