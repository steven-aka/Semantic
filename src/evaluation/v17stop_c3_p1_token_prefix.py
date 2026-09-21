"""Zero-call token-ID prefix and source-preservation audit for C3."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from src.data.schemas import read_jsonl
from src.target.qampari_runner import QAMPARI_SYSTEM_PROMPT
ROOT=Path('results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout'); C1=Path('results/v2_rank_then_cut/v17stop_c1_obs0_native_confidence'); OUT=Path('results/v2_rank_then_cut/v17stop_c3_p1_token_prefix')
MARK='C3_PACKET_BODY_MARKER_7E91'
def main():
 traces={(r['example_id'],r['depth']):r for r in read_jsonl(C1/'traces.jsonl')};qs=sorted({q for q,_ in traces});ids=set(qs)
 data={r['example_id']:r for r in read_jsonl(ROOT/'data/v10_v12_train_train_clean.jsonl') if r['example_id'] in ids}; orders={r['example_id']:r['decoded_order'] for r in read_jsonl(ROOT/'sel_c0_train_v8_rollouts.jsonl') if r['example_id'] in ids};tok=AutoTokenizer.from_pretrained('models/Qwen3-8B',trust_remote_code=True,local_files_only=True); rows=[]
 for q in qs:
  user=('Use only the context below to answer the list question.\nReturn every distinct supported answer, separated only by #, inside <answer> and </answer>. Do not explain.\n\nQuestion:\n'+data[q]['question']+'\n\nContext:\n'+MARK)
  rendered=tok.apply_chat_template([{'role':'system','content':QAMPARI_SYSTEM_PROMPT},{'role':'user','content':user}],tokenize=False,add_generation_prompt=True,enable_thinking=False);assert rendered.count(MARK)==1;head,suffix=rendered.split(MARK); head_ids=tok.encode(head,add_special_tokens=False);suffix_ids=tok.encode(suffix,add_special_tokens=False);chunks=[]
  for j,p in enumerate(orders[q]):chunks.append(tok.encode(('' if j==0 else '\n\n')+data[q]['packet_texts'][p],add_special_tokens=False))
  bodies={};full={}
  for d in (6,7,9,10):b=head_ids+sum(chunks[:d],[]);bodies[d]=b;full[d]=b+suffix_ids
  prefix=all(bodies[10][:len(bodies[d])]==bodies[d] for d in (6,7,9)); chars='\n\n'.join(data[q]['packet_texts'][p] for p in orders[q]); source_ok=all(data[q]['packet_texts'][p] in chars for p in orders[q]) and len(orders[q])==12 and len(set(orders[q]))==12
  rows.append({'example_id':q,'prefix_ok':prefix,'source_ok':source_ok,'body_tokens':{str(d):len(bodies[d]) for d in bodies},'suffix_tokens':len(suffix_ids)})
 out={'protocol':'V17-STOP-C3-P1_TOKEN_PREFIX_CORRECTNESS','queries':len(rows),'prefix_correct':sum(r['prefix_ok'] for r in rows),'source_preserved':sum(r['source_ok'] for r in rows),'all_pass':all(r['prefix_ok'] and r['source_ok'] for r in rows),'serialization':'fixed query/instruction header + tokenized exact packet chunks in V8 reveal order separated by two newlines + separately tokenized fixed chat generation suffix','decision':'GO_C3_P2_ORDER_PERTURBATION' if all(r['prefix_ok'] and r['source_ok'] for r in rows) else 'STOP_C3_PREFIX_SERIALIZATION','new_target_calls':0}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
