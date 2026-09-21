"""Fresh paired depth10 quality test: canonical block order vs V8 reveal order."""
import hashlib,json
from pathlib import Path
from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction,qampari_list_metrics
from src.target.qampari_runner import QampariTargetRunner,QAMPARI_SYSTEM_PROMPT,build_qampari_prompt
ROOT=Path('results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout');C1=Path('results/v2_rank_then_cut/v17stop_c1_obs0_native_confidence');P0=Path('results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain');OUT=Path('results/v2_rank_then_cut/v17stop_c3_p2_order_quality');MARK='C3_PACKET_BODY_MARKER_7E91';LEVELS=(.6,.7,.8,.9,.95)
def main():
 from vllm import SamplingParams,TokensPrompt
 traces={(r['example_id'],r['depth']):r for r in read_jsonl(C1/'traces.jsonl')};pop={q for q,_ in traces};qs=sorted(pop,key=lambda q:hashlib.sha256(('C3-P2|'+q).encode()).hexdigest())[:128];ids=set(qs);data={r['example_id']:r for r in read_jsonl(ROOT/'data/v10_v12_train_train_clean.jsonl') if r['example_id'] in ids};orders={r['example_id']:r['decoded_order'] for r in read_jsonl(ROOT/'sel_c0_train_v8_rollouts.jsonl') if r['example_id'] in ids};atoms={r['example_id']:r['answer_atoms'] for r in read_jsonl('data/units/qampari_rank_v2_candidates5000_annotations.jsonl') if r['example_id'] in ids};elig={r['example_id']:{float(k) for k,v in r['levels'].items() if v is not None} for r in read_jsonl(P0/'per_query.jsonl') if r['example_id'] in ids}
 target=QampariTargetRunner('models/Qwen3-8B',backend='vllm',max_new_tokens=256,gpu_memory_utilization=.55,max_model_len=4096);tok=target.tokenizer;prompts=[];meta=[]
 for q in qs:
  chosen=orders[q][:10]; canonical='\n\n'.join(x.strip() for i,x in enumerate(data[q]['packet_texts']) if i in set(chosen)); cp=target._chat_prompts([data[q]['question']],[canonical])[0];prompts.append(TokensPrompt(prompt_token_ids=tok.encode(cp)));meta.append((q,'canonical'))
  user=build_qampari_prompt(data[q]['question'],MARK); rendered=tok.apply_chat_template([{'role':'system','content':QAMPARI_SYSTEM_PROMPT},{'role':'user','content':user}],tokenize=False,add_generation_prompt=True,enable_thinking=False);head,suffix=rendered.split(MARK);ids2=tok.encode(head,add_special_tokens=False)
  for j,p in enumerate(chosen):ids2+=tok.encode(('' if j==0 else '\n\n')+data[q]['packet_texts'][p],add_special_tokens=False)
  ids2+=tok.encode(suffix,add_special_tokens=False);prompts.append(TokensPrompt(prompt_token_ids=ids2));meta.append((q,'reveal'))
 outs=target.model.generate(prompts,SamplingParams(temperature=0,max_tokens=256),use_tqdm=False);rows=[]
 for (q,arm),o in zip(meta,outs):
  ans=parse_list_prediction(o.outputs[0].text);rows.append({'example_id':q,'arm':arm,'prediction':' # '.join(ans),'f1':float(qampari_list_metrics(ans,atoms[q])['f1']),'prompt_tokens':len(o.prompt_token_ids),'generated_tokens':len(o.outputs[0].token_ids)})
 by={(r['example_id'],r['arm']):r for r in rows};summary={}
 for arm in ('canonical','reveal'):
  suc={str(t):sum(by[(q,arm)]['f1']+1e-12>=t for q in qs if t in elig[q]) for t in LEVELS};complete=sum(all(by[(q,arm)]['f1']+1e-12>=t for t in elig[q]) for q in qs);summary[arm]={'success':suc,'complete':complete,'mean_prompt_tokens':sum(by[(q,arm)]['prompt_tokens'] for q in qs)/len(qs),'mean_generated_tokens':sum(by[(q,arm)]['generated_tokens'] for q in qs)/len(qs)}
 tol=1.28;quality=all(summary['reveal']['success'][str(t)]>=summary['canonical']['success'][str(t)]-tol for t in LEVELS) and summary['reveal']['complete']>=summary['canonical']['complete']-tol;out={'protocol':'V17-STOP-C3-P2_ORDER_PERTURBATION','queries':128,'single_variable':'packet block order only; canonical chat template and question-after-context suffix preserved','results':summary,'quality_gate_pass':quality,'decision':'GO_C3_P3_REAL_CACHE_MICROBENCH' if quality else 'STOP_C3_ORDER_PERTURBATION','target_calls':256,'sealed_sets_read':False};OUT.mkdir(parents=True,exist_ok=True);(OUT/'summary.json').write_text(json.dumps(out,indent=2)+'\n');open(OUT/'paired.jsonl','w').write(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
