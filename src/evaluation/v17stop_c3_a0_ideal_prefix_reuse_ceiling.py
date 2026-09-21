"""Zero-call ideal prefix-reuse compute ceiling for frozen C1 B1=0.95."""
import json
from pathlib import Path
from src.data.schemas import read_jsonl
ROOT=Path('results/v2_rank_then_cut/v17stop_c1_obs0_native_confidence')
LEVELS=(.6,.7,.8,.9,.95); ED={.6:6,.7:7,.8:7,.9:9,.95:10}; TH=.95
def main():
 states={(r['example_id'],int(r['depth'])):r for r in read_jsonl(ROOT/'traces.jsonl')}; scores={(r['example_id'],float(r['tau'])):r for r in read_jsonl(ROOT/'oof_scores.jsonl')};qs=sorted({q for q,_ in states}); base=ideal=ctx=0; stops=fallback=0
 for q in qs:
  for t in LEVELS:
   e=states[(q,ED[t])];l=states[(q,10)];base+=l['prompt_tokens']+l['generated_tokens']
   stop=True if t==.95 else scores[(q,t)]['B1_native_only']>=TH
   if stop:
    ideal+=e['prompt_tokens']+e['generated_tokens'];ctx+=e['context_tokens'];stops+=1
   else:
    # Ideal prefixable contract: depth10 prefill is computed once in total;
    # the early observation adds only its decode tokens.
    ideal+=l['prompt_tokens']+e['generated_tokens']+l['generated_tokens'];ctx+=l['context_tokens'];fallback+=1
 base/=len(qs);ideal/=len(qs);ctx/=len(qs);saving=base-ideal
 out={'protocol':'V17-STOP-C3-A0_IDEAL_PREFIX_REUSE_COMPUTE_CEILING','queries':len(qs),'policy':'frozen C1 B1 native-only threshold 0.95','mean_depth10_compute_tokens_per_query':base,'mean_ideal_prefix_reuse_compute_tokens_per_query':ideal,'mean_compute_tokens_saved_per_query':saving,'compute_saving_fraction':saving/base,'passes_2pct_gate':ideal<=.98*base,'mean_final_context_tokens_per_query':ctx,'stop_requests':stops,'fallback_requests':fallback,'decision':'GO_C3_PREFIX_TOKEN_PREFLIGHT' if ideal<=.98*base else 'STOP_C3_BEFORE_TARGET','assumptions':['100% reuse of evidence-prefix KV','zero cache-management overhead','fallback pays depth10 prompt prefill once plus both early and depth10 decode','frozen C1 OOF decisions; no refitting'],'new_target_calls':0,'sealed_sets_read':False}
 outdir=Path('results/v2_rank_then_cut/v17stop_c3_a0_ideal_prefix_reuse_ceiling');outdir.mkdir(parents=True,exist_ok=True);(outdir/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
