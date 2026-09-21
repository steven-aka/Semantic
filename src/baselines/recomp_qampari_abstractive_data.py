"""Prepare RECOMP-style teacher-distillation inputs without loading the Target."""
from __future__ import annotations

import argparse, json
from pathlib import Path
from src.data.schemas import read_jsonl


PROMPT = """Create a concise, source-grounded evidence summary for the list question.
Include every distinct answer-bearing fact supported by the documents. Preserve names and
the relation asked by the question. Do not answer as a bare list, add outside facts, or
mention these instructions. Return at most 192 tokens.

Question: {question}

Documents:
{context}

Evidence summary:"""


def render(row):
    return "\n\n".join(row["packet_texts"])


def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--limit",type=int)
    a=p.parse_args(); a.output.parent.mkdir(parents=True,exist_ok=True)
    rows=list(read_jsonl(a.source)); rows=rows[:a.limit] if a.limit else rows
    with a.output.open("w") as f:
        for r in rows:
            context=render(r)
            out={"example_id":r["example_id"],"question":r["question"],"context":context,
                 "teacher_prompt":PROMPT.format(question=r["question"],context=context)}
            f.write(json.dumps(out,ensure_ascii=False)+"\n")
    print(json.dumps({"rows":len(rows),"output":str(a.output),"target_loaded":False}))

if __name__=="__main__": main()
