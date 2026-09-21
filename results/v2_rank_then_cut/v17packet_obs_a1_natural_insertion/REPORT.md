# OBS-A1: natural protected-insertion cohort

We froze a deployment-visible action before Target calls: at V8 depth 9, preserve the default context and insert the shortest exact title-plus-sentence fragment from rank 10 with 0–48 additional Qwen3-8B context tokens. From 781 train-side P0 queries unused by R1 and the earlier title-only pilot, 500 were eligible; SHA order selected 256 without looking at baseline quality or insertion outcome. Baseline and action were paired in the same vLLM run under the canonical Qwen3-8B contract. This is training-side design data, not independent confirmation.

| Depth-9 policy | 0.90 success | Mean added context tokens |
|---|---:|---:|
| V8 baseline | 180/256 | 0 |
| Fixed shortest protected insertion | 186/256 | 26.64 |

The paired change is 27 repairs and 21 breaks. Mean continuous F1 changes by +0.01394: 109 actions gain F1, 39 lose it, and 108 tie. A query bootstrap gives a 95% interval of **−7 to +20** for the net count of 0.90 successes and **−0.00362 to +0.03113** for mean F1 change. Neither interval excludes zero. Target token use increases by 28.93 per query on average. Historical P0 V8 depth 10 succeeds on 216/256 and uses 92.70 more context tokens per query than the inserted depth-9 context; it was not freshly paired in this run and is only a discrete reference point. No five-anchor or Complete Pareto claim follows from this depth-9 experiment.

To check categorical repeatability, we selected SHA-first 10 original repairs, 10 breaks and 10 neutral cases, then reran baseline and action together. All 30 repeated their original repair/break/neutral category. This is reassuring for these selected cases, but the conditional sampling and small size do not establish a population noise ceiling.

**Decision:** the action space offers real local opportunities, but uniform insertion is not yet a credible deployable improvement: it increases token use, produces 21 breaks, and has uncertain net quality gain. The repeat result reduces concern that those specific break/repair labels are merely unstable. A strong paired-effect probe can now use 256 independent, outcome-blind queries as a *diagnostic*, but has only 27 repair and 21 break queries; it cannot serve as an information-theoretic ceiling. Before any final policy claim, candidate choice and the decision to intervene must be tested through query-held-out rollouts against V8 across all five anchors and cost axes. Do not combine these insertion labels with R1 replacement labels or use the sealed sets for tuning.

[Frozen protocol](../../../configs/v17packet_obs_a1_natural_insertion.json), [runner](../../../src/evaluation/v17packet_obs_a1_natural_insertion.py), [analysis](../../../src/evaluation/v17packet_obs_a1_analyze.py), [paired summary](summary.json), [bootstrap and depth-10 reference](analysis.json), and [repeat summary](repeat_summary.json) retain the evidence.
