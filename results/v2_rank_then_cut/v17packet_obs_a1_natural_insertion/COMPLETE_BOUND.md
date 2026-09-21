# OBS-A3: final-objective value bound for depth-9 insertion

We projected the OBS-A1 fresh paired 0.90 results onto the already frozen five-level stop schedule `[6, 7, 7, 9, 10]`. The other four levels use the unchanged V8 prefixes from the earlier P0 Qwen3-8B cache. An unattainable anchor is undefined, not a failure. This is a **hybrid-cache projection** because the other levels were not rerun in the OBS-A1 batch; it is not an independent end-to-end confirmation.

| Policy on 256 natural train queries | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 | Complete |
|---|---:|---:|---:|---:|---:|---:|
| V8 | 223 | 222 | 172 | 180 | 173 | 129 |
| Uniform shortest protected insertion at depth 9 | 223 | 222 | 172 | 186 | 173 | 127 |

Uniform insertion produces nine Complete repairs and eleven Complete breaks, so Complete falls by two despite six net 0.90 successes. Only **9 of the 27** 0.90 repairs have all other attainable levels passing. Among the other 18, earlier 0.80 fails in 15 cases, 0.60 in seven, 0.70 in five, and later 0.95 in four; failures overlap. The depth-9 action cannot change those earlier or later stop contexts. This is a structural limit of this exact action and schedule, independent of the strength of an insertion gate.

A hindsight selector that inserts only on the nine Complete-repair queries would move Complete 129→138/256 at just +0.90 mean context tokens/query. A hindsight selector targeting all 27 0.90 repairs moves 0.90 180→207/256 but only moves Complete 129→138; it costs +2.75 mean context tokens/query. Both use forbidden Target outcomes and are oracle bounds, not deployable results. The complete-repair opportunity prevalence is only 9/256, making a trainable safe controller substantially harder than the 27/256 0.90-repair prevalence suggests.

The earlier P0 chain itself shows why 0.80 matters: at fixed depth 7 it passes on 172/256; depth 8 passes on 208/256 at approximately +83 mean context tokens. This is a historical discrete schedule comparison, not a new policy selection.

**Decision:** `STOP_DEPTH9_INSERTION_LABEL_SCALE_AND_STRONG_PROBE_FOR_COMPLETE`. The problem is not merely that the cheap gate fails. This one-slot action affects only the 0.90 point, while Complete failures often occur at the earlier 0.80 point. Adding labels or a larger model for this exact action would primarily optimize a narrow surrogate, not the five-anchor compression objective. A future action must first demonstrate a credible multi-anchor quality–token ceiling under a frozen schedule, including 0.80 and 0.95, before learning a selector. Keep the fixed V8 Pareto frontier as the deployable reference; do not open sealed sets or claim the hybrid numbers as final validation.

[Code](../../../src/evaluation/v17packet_obs_a3_complete_bound.py), [summary](complete_bound.json), and [per-query projection](complete_bound_per_query.jsonl) support this decision.
