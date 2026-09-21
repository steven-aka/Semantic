# OBS-A0: protected-insertion data gate

This is a read-only audit of the existing Qwen3-8B train-side cache. The action is fixed V8 depth-9 context plus one rank-10 title-and-sentence fragment, preserving V8's default evidence, with at most 48 extra context tokens. It must not be pooled with R1's budget-neutral replacement labels.

The SLOT-B0 pilot contains 84 distinct candidate actions on only 24 independent queries: 16 0.90 repairs, five breaks and 63 actions that leave the 0.90 bit unchanged. The 16 repairs occur on seven queries and all five breaks on two queries. Sampling deliberately balanced 12 baseline successes and 12 failures within an earlier design-exposed frame, so these rates are not natural-population estimates. Treating 84 actions as 84 independent observations would inflate evidence for learnability.

FRAG-A0 repeated 27 outcome-selected actions on 15 queries. The fresh baseline reproduced the cached 0.90 bit in 27/27 cases; the title-plus-sentence action reproduced it in 26/27, including 15/16 selected repairs and 5/5 selected breaks. This supports limited execution consistency but cannot estimate the population label-noise ceiling. No fresh Target calls were made for this gate.

**Decision:** `STOP_STRONG_OBSERVABILITY_PROBE_ON_CURRENT_INSERTION_LABELS`. A stronger encoder trained on these 24 queries would measure overfitting to a small, stratified sample, not whether deployment-visible text contains a generalizable effect signal. R2G0's 512-query labels are for a different replacement action, and cannot fill this gap.

The minimal next label collection is an outcome-blind, query-level train-side cohort using one frozen, deployment-visible insertion candidate rule. Run matched baseline and action calls in the same Target execution contract, save raw outputs, parsed answers, continuous F1, five-level results and real context/Target cost. Include a small preregistered repeat subset spanning naturally observed gain/loss/neutral cases. Only then run one query-grouped paired-effect probe; evaluate actual repairs, breaks and quality–token tradeoffs alongside effect correlation. A failed probe would reject that specified model and data protocol, not prove an information-theoretic observability limit.

The reproducible counts are in [summary.json](summary.json) and [the audit script](../../../src/evaluation/v17packet_obs_a0_data_gate.py).
