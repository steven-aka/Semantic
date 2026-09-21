# OBS-A2: grouped paired-text lexical control

This zero-Target-call diagnostic uses the 256 natural protected-insertion queries from OBS-A1. The input contains the full question, V8 depth-9 baseline context and the actual inserted context, with explicit field markers and no model token truncation. Within each of four example-hash folds, TF–IDF unigram/bigram features and one fixed ridge model predict continuous action-minus-baseline F1. The vocabulary is fit on that fold's training queries only. Fold-internal score ranks are used for fixed 5% and 10% intervention budgets, avoiding cross-fold intercept scale comparisons. No hyperparameter sweep or sealed data were used.

| Diagnostic | Result |
|---|---:|
| OOF Spearman, predicted vs actual ΔF1 | 0.1045 |
| OOF repair-vs-break AUC on 48 decisive queries | 0.5291 |
| Top 5% interventions | 0 repair, 1 break; random expects 1.37 repair, 1.07 break |
| Top 10% interventions | 2 repair, 2 break; random expects 2.74 repair, 2.13 break |

The folds are unstable: fold 0 has only two repairs and nine breaks (within-fold AUC 0.00), while folds 1–3 have AUC 0.64–0.78 on small decisive subsets. The pooled AUC mixes fold score scales and is descriptive; intervention budgets use fold-internal ranks. The simple lexical paired representation has no useful low-budget safety/benefit separation. This result is consistent with the earlier R2G0 failure under a different replacement action, but does not establish that full deployment-visible text lacks usable semantic information. There are only 27 repair and 21 break queries in the current natural insertion cohort; stronger-model failure on this set would have wide uncertainty and would not be an information-theoretic ceiling. Conversely, 30/30 categorical repeatability in the separate outcome-stratified subset argues against dismissing all break/repair outcomes as transient execution noise.

**Decision:** `STOP_CHEAP_PAIRED_TEXT_GATE`. Do not deploy this control, tune its threshold, or spend sealed sets on it. The remaining research choice is whether the cost of gathering more independent paired insertion labels and testing one frozen semantic probe is justified. Because uniform insertion gives only +6 net 0.90 successes with 21 breaks and +26.64 tokens/query, that investment should require a preregistered cost and effect-size gate. At present, the best supported deployable policy in this branch remains V8 STAY.

[Code](../../../src/evaluation/v17packet_obs_a2_paired_text_control.py), [summary](paired_text_control_summary.json) and [OOF scores](paired_text_control_oof.jsonl) reproduce the diagnostic.
