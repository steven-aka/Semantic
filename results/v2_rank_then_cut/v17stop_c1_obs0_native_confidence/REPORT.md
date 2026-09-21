# C1-OBS0 native-confidence two-stage stopping

## Cohort and contract

We collected 1,024 same-call traced states (`d6/d7/d9/d10`) for 256 naturally selected train-side queries, excluding all 32 prior trace-preflight IDs. Answers, F1 labels, token log-probabilities and real prompt/generation costs came from the same traced execution path. These queries are design-exposed; no sealed split was read.

## Baselines and oracle

| Policy | 0.60/0.70/0.80/0.90/0.95 | Complete | Final context/query | Target compute/query |
|---|---|---:|---:|---:|
| Aggressive fixed | 223/222/189/186/183 | 135 | 2316.96 | 3123.73 |
| Depth10 fixed | 245/244/242/220/183 | 183 | 3325.51 | 4185.21 |
| Two-stage oracle | 250/248/245/230/183 | 183 | 2510.70 | 3826.16 |

## OOF result

No output-only, native-only, or combined point passes all quality, context-capture, and real Target-compute gates. The formal decision is `STOP_C1_NATIVE_CONFIDENCE`.

The informative diagnostic is native-only threshold `0.95`: it yields five-anchor successes `245/245/243/219/183`, Complete `182`, captures `33.1%` of oracle context saving, and uses `3055.59` final-context tokens/query. It nevertheless uses `5725.41` Target tokens/query, `1540.20` more than direct depth10.

Thus token-level confidence contains some quality/context signal that B0 lacks, but repeated full inference prevents deployment Pareto improvement. B2 did not pass, so this is not a successful stopper claim. It authorizes only `C2-COST_PREFIX_STATE_REUSE_PREFLIGHT`: verify whether the actual prompt/KV execution can reuse prefix state without changing text, answers, or costs before implementing any serving optimization.
