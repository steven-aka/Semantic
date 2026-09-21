# V17 STOP-C0 Target-feedback gate reassessment

The proposed feedback-stopping line was evaluated from the existing fresh canonical V8 prefix cache; no new Target calls and no model fitting were needed.

## Primary prescribed baseline

The frozen schedule is `[6, 7, 7, 9, 10]`. Its mean cumulative context cost is `2183.62` tokens/query. A free clairvoyant stopper lowers this only to `2084.92`, an optimistic saving of `98.69` tokens/query or `4.52%`. This fails the preregistered `>=5%` gate before charging any observation call.

## Earlier-success opportunity

| Fidelity | Fixed depth | Eligible | Earlier success | Fraction | >=10% |
|---|---:|---:|---:|---:|---:|
| 0.6 | 6 | 1421 | 1134 | 79.80% | yes |
| 0.7 | 7 | 1421 | 1051 | 73.96% | yes |
| 0.8 | 7 | 1421 | 3 | 0.21% | no |
| 0.9 | 9 | 1421 | 1 | 0.07% | no |
| 0.95 | 10 | 1131 | 1 | 0.09% | no |


Only `2/5` anchors pass the 10% opportunity requirement; the protocol requires at least `3/5`. The 0.80, 0.90 and 0.95 operating depths coincide with the sharp empirical quality transitions, so output feedback cannot create substantial earlier context opportunities there.

The existing depth-10 quality-heavy control remains positive (sequential oracle saves `756.73` final-context tokens and `594.30` Target-compute tokens/query), but it is a different operating point and does not make the prescribed early-schedule gate pass.

## Decision

`STOP_CURRENT_CONTRACT_ADAPTIVE_STOPPING_AT_C0`.

Do not repeat answer-dynamics probes or train another output-only stopper. The next scientifically distinct contract is `C1-OBS_NATIVE_TARGET_CONFIDENCE`: expose read-only native token log-probability or margin traces from the frozen Target, while retaining source-preserving text, V8 ordering, add-only prefixes and the fidelity objective. Any pilot must collect outcomes and traces in the same fresh calls; prior trace preflight showed that attaching new traces to old cached generations is invalid.
