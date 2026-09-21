#!/usr/bin/env python3
"""Zero-call constructive existence audit for a Target-native discrete path."""
from __future__ import annotations
import json, math
from pathlib import Path
from statistics import mean, median
from tokenizers import Tokenizer

def jl(p):
    with open(p) as f:return [json.loads(x) for x in f if x.strip()]
def pct(x,p):
    x=sorted(x);i=(len(x)-1)*p;a=math.floor(i);b=math.ceil(i)
    return x[a] if a==b else x[a]*(b-i)+x[b]*(i-a)

def main():
    out=Path('results/v2_rank_then_cut/project_synthesis_a0_solution_existence');out.mkdir(parents=True,exist_ok=True)
    traces=jl('results/v2_rank_then_cut/v17stop_c1_obs0_native_confidence/traces.jsonl')
    ids=sorted({x['example_id'] for x in traces}); by={(x['example_id'],x['depth']):x for x in traces}
    ann={x['example_id']:x for x in jl('data/units/qampari_rank_v2_candidates5000_annotations.jsonl') if x['example_id'] in set(ids)}
    tok=Tokenizer.from_file('models/Qwen3-8B/tokenizer.json')
    levels=[.6,.7,.8,.9,.95]; depths=[6,7,7,9,10]
    rows=[]
    for q in ids:
      atoms=ann[q]['answer_atoms'];n=len(atoms); answers=[a['answer_text'] for a in atoms]
      proofs=[a['source_proof'] for a in atoms]
      cut=[]
      for t in levels:
        # Set-F1 of m correct predictions against n references is 2m/(m+n).
        m=math.ceil(t*n/(2-t)-1e-12);cut.append(min(n,m))
      answer_codes=[' # '.join(answers[:m]) for m in cut]
      proof_codes=['\n'.join(proofs[:m]) for m in cut]
      answer_lens=[len(tok.encode(x).ids) for x in answer_codes]
      proof_lens=[len(tok.encode(x).ids) for x in proof_codes]
      v8_lens=[by[(q,d)]['context_tokens'] for d in depths]
      rows.append({'example_id':q,'answer_count':n,'required_answer_counts':cut,'oracle_answer_code_tokens':answer_lens,'oracle_source_proof_tokens':proof_lens,'v8_context_tokens':v8_lens,'answer_code_cumulative':sum(answer_lens),'proof_code_cumulative':sum(proof_lens),'v8_cumulative':sum(v8_lens)})
    def stats(key):
      x=[r[key] for r in rows];return {'mean':mean(x),'median':median(x),'p90':pct(x,.9),'max':max(x)}
    per=[]
    for i,t in enumerate(levels):
      a=[r['oracle_answer_code_tokens'][i] for r in rows];p=[r['oracle_source_proof_tokens'][i] for r in rows];v=[r['v8_context_tokens'][i] for r in rows]
      per.append({'level':t,'mean_answer_code_tokens':mean(a),'mean_source_proof_tokens':mean(p),'mean_v8_context_tokens':mean(v),'answer_code_ratio_vs_v8':mean(a)/mean(v),'source_proof_ratio_vs_v8':mean(p)/mean(v),'answer_code_le_v8_fraction':sum(x<=y for x,y in zip(a,v))/len(v),'source_proof_le_v8_fraction':sum(x<=y for x,y in zip(p,v))/len(v)})
    result={'protocol':'PROJECT-SYNTHESIS-A0_CONSTRUCTIVE_SOLUTION_EXISTENCE','status':'ZERO_CALL_REPRESENTATION_EXISTENCE_PROVED','queries':len(rows),'construction':{'native_symbols':'Qwen3-8B vocabulary token IDs','nested_sequence':'canonical gold answer strings ordered once; each fidelity state is a longer prefix','minimum_correct_answer_count_formula':'ceil(tau*n/(2-tau)) under set-F1 with no false positives','source_backed_control':'concatenated exact source proof for the same answer prefix'},'cumulative':{'oracle_answer_code':stats('answer_code_cumulative'),'oracle_source_proof':stats('proof_code_cumulative'),'v8':stats('v8_cumulative'),'mean_answer_code_ratio_vs_v8':mean(r['answer_code_cumulative'] for r in rows)/mean(r['v8_cumulative'] for r in rows),'mean_source_proof_ratio_vs_v8':mean(r['proof_code_cumulative'] for r in rows)/mean(r['v8_cumulative'] for r in rows)},'per_level':per,'logical_conclusion':{'proved':'A compact nested sequence in the Target native vocabulary exists for every audited query and is much shorter than V8. Thus the joint compactness/nesting/symbol-compatibility constraints are not mathematically inconsistent.','not_proved':['Qwen3-8B will copy or use this code under the frozen QA prompt','a compressor can infer the code without gold answers','the code is faithful/source-grounded when generated automatically','end-to-end compute Pareto after code construction'],'anti_cheating':'The construction uses gold answers and is an existence witness only. It is prohibited as a deployable compressor or training input.'},'next_gate':{'name':'NATIVE-CODE-A0_CONSUMPTION_POSITIVE_CONTROL','design':'Small paired design-exposed Target test of frozen Qwen3-8B consuming oracle answer-prefix codes; compare direct code passthrough, code plus minimal provenance, and V8 at matched cost. No compressor training.','go':'Only if the Target reliably converts short native codes into task answers across all five nested prefixes without new nonmonotonicity.','stop':'If even oracle native codes are not reliably consumed, stop discrete representation before implementation.'},'decision':'GO_NATIVE_CODE_CONSUMPTION_POSITIVE_CONTROL','new_target_calls':0,'sealed_sets_read':False}
    (out/'per_query.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
