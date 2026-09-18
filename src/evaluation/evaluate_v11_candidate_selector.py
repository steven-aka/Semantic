from __future__ import annotations
import argparse,json
from pathlib import Path
from statistics import mean
from src.data.schemas import read_jsonl,write_jsonl
from src.evaluation.v10_mask_value_decision import decide_v10_probe
from src.reproducibility import sha256,write_metadata

def main():
 p=argparse.ArgumentParser();p.add_argument('--selection',required=True);p.add_argument('--data',required=True);p.add_argument('--v8-details',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 selections={r['example_id']:r for r in read_jsonl(a.selection)};rows={r['example_id']:r for r in read_jsonl(a.data)};base={r['example_id']:r for r in read_jsonl(a.v8_details)}
 if set(selections)!=set(rows) or set(base)!=set(rows):raise ValueError('ID mismatch')
 per={l:[] for l in [.6,.7,.8,.9,.95]};details=[];complete=0;regrets=[];anchors=successes=0
 for eid,row in rows.items():
  selected=selections[eid];contracts=int(selected['outcome']['contracts']);active=len(row['attainable_levels']);all_ok=contracts==active;complete+=all_ok
  oracle=sum(int(x['oracle_nested_tokens']) for x in base[eid]['anchors']);full=int(row['mask_values'][4095]['tokens']);regret=(int(selected['outcome']['cumulative_tokens'])-oracle)/(active*full) if all_ok else None
  if regret is not None:regrets.append(regret)
  ar=[]
  for i,level in enumerate(row['attainable_levels'],1):
   ok=contracts>=i;successes+=ok;anchors+=1;per[float(level)].append(ok);ar.append({'fidelity_level':level,'contract_success':ok})
  details.append({**selected,'all_active_contracts_success':all_ok,'oracle_cutoff_ranking_regret_normalized':regret,'anchors':ar})
 summary={'complete':True,'examples':len(rows),'oracle_cutoff_active_contract_success_fraction':successes/anchors,'oracle_cutoff_all_active_contracts_success_fraction':complete/len(rows),'oracle_cutoff_feasible_trajectory_examples':len(regrets),'mean_oracle_cutoff_ranking_regret_normalized_feasible':mean(regrets) if regrets else None,'per_level':{str(l):{'examples':len(v),'contract_successes':sum(v),'contract_success_fraction':mean(v)} for l,v in per.items()},'decoder':'V8 fallback plus frozen V10 top16 mask projections selected by V11 candidate head','artifacts':{'selection_sha256':sha256(a.selection),'data_sha256':sha256(a.data),'v8_details_sha256':sha256(a.v8_details)}}
 write_jsonl(a.output,details);write_metadata(str(Path(a.output).with_suffix('.summary.json')),summary);write_metadata(str(Path(a.output).with_suffix('.decision.json')),decide_v10_probe(summary));print(json.dumps(summary,indent=2));print(json.dumps(decide_v10_probe(summary),indent=2))
if __name__=='__main__':main()
