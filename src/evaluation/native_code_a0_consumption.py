#!/usr/bin/env python3
"""Frozen-Qwen3 consumption positive control for oracle native answer codes."""
from __future__ import annotations
import json, math
from pathlib import Path
from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.target.qampari_runner import QampariTargetRunner

LEVELS=[.6,.7,.8,.9,.95]
def sf1(a,b):
 x={z.strip().lower() for z in a if z.strip()};y={z.strip().lower() for z in b if z.strip()}
 if not x or not y:return float(x==y)
 p=len(x&y)/len(x);r=len(x&y)/len(y);return 2*p*r/(p+r) if p+r else 0.

def main():
 out=Path('results/v2_rank_then_cut/native_code_a0_consumption');out.mkdir(parents=True,exist_ok=True)
 traces=[x for x in read_jsonl('results/v2_rank_then_cut/v17stop_c1_obs0_native_confidence/traces.jsonl') if x['depth']==10]
 ids=sorted(x['example_id'] for x in traces)[:32]; ids_set=set(ids)
 data={x['example_id']:x for x in read_jsonl('results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/data/v10_v12_train_train_clean.jsonl') if x['example_id'] in ids_set}
 ann={x['example_id']:x for x in read_jsonl('data/units/qampari_rank_v2_candidates5000_annotations.jsonl') if x['example_id'] in ids_set}
 tasks=[]; direct=[]
 for qi,q in enumerate(ids):
  atoms=ann[q]['answer_atoms'];answers=[a['answer_text'] for a in atoms];n=len(answers)
  dq=ids[(qi+1)%len(ids)];danswers=[a['answer_text'] for a in ann[dq]['answer_atoms']]
  for t in LEVELS:
   m=min(n,math.ceil(t*n/(2-t)-1e-12)); code=answers[:m]
   # Match item count, cycling the next query if needed.
   decoy=(danswers*((m+len(danswers)-1)//len(danswers)))[:m]
   for arm,items in [('correct',code),('shuffled',decoy)]:
    context=('Compressed task-relevant answer code. Each item is a candidate answer; '
             'return all and only the items that answer the question.\nCODE: '+' # '.join(items))
    tasks.append({'example_id':q,'level':t,'arm':arm,'question':data[q]['question'],'context':context,'code_items':items})
   direct.append({'example_id':q,'level':t,'direct_code_gold_f1':qampari_list_metrics(code,atoms)['f1'],'code_items':code})
 target=QampariTargetRunner('models/Qwen3-8B',backend='vllm',max_new_tokens=256,gpu_memory_utilization=.55,max_model_len=4096)
 records=target.generate_batch_records([x['question'] for x in tasks],[x['context'] for x in tasks])
 rows=[]
 for task,rec in zip(tasks,records):
  pred=parse_list_prediction(rec['text']); atoms=ann[task['example_id']]['answer_atoms']
  rows.append({**task,'raw':rec['text'],'prediction':pred,'code_recovery_f1':sf1(pred,task['code_items']),'gold_f1':qampari_list_metrics(pred,atoms)['f1'],'prompt_tokens':len(target.tokenizer.encode(target._chat_prompts([task['question']],[task['context']])[0])),'generated_tokens':rec['generated_tokens']})
 correct=[x for x in rows if x['arm']=='correct']; shuffled=[x for x in rows if x['arm']=='shuffled']
 mean=lambda xs:sum(xs)/len(xs)
 per=[]
 for t in LEVELS:
  c=[x for x in correct if x['level']==t];s=[x for x in shuffled if x['level']==t]
  retention=[]
  if t!=LEVELS[0]:
   prev=LEVELS[LEVELS.index(t)-1]
   pc={x['example_id']:x for x in correct if x['level']==prev}
   for x in c:
    old=set(z.lower() for z in pc[x['example_id']]['code_items']);got=set(z.lower() for z in x['prediction']);retention.append(len(old&got)/len(old) if old else 1)
  per.append({'level':t,'correct_code_recovery_f1':mean([x['code_recovery_f1'] for x in c]),'correct_gold_f1':mean([x['gold_f1'] for x in c]),'shuffled_gold_f1':mean([x['gold_f1'] for x in s]),'decoy_sensitivity_gap':mean([x['gold_f1'] for x in c])-mean([x['gold_f1'] for x in s]),'nested_prior_code_retention':mean(retention) if retention else None})
 avg_recovery=mean([x['code_recovery_f1'] for x in correct]);avg_ret=mean([x['nested_prior_code_retention'] for x in per if x['nested_prior_code_retention'] is not None]);gap=mean([x['decoy_sensitivity_gap'] for x in per]);direct_mean=mean([x['direct_code_gold_f1'] for x in direct]);target_gold=mean([x['gold_f1'] for x in correct])
 gates={'code_recovery_ge_0.95':avg_recovery>=.95,'nested_retention_ge_0.95':avg_ret>=.95,'decoy_gap_ge_0.50':gap>=.50}
 summary={'protocol':'NATIVE-CODE-A0_CONSUMPTION_POSITIVE_CONTROL','queries':32,'target_calls':len(tasks),'per_level':per,'aggregate':{'code_recovery_f1':avg_recovery,'nested_retention':avg_ret,'correct_minus_shuffled_gold_f1':gap,'direct_code_gold_f1':direct_mean,'target_correct_code_gold_f1':target_gold,'target_value_over_direct':target_gold-direct_mean},'gates':gates,'decision':'GO_NATIVE_CODE_CONTRACT_REVIEW' if all(gates.values()) else 'STOP_NATIVE_CODE_TARGET_COMPATIBILITY','interpretation':'Positive control only. Oracle answer items are leakage and cannot be compressor inputs. Target value over direct code determines whether this remains compression or collapses to direct QA.','sealed_sets_read':False}
 (out/'rows.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
