# C1-C0B two-stage oracle ceiling

This zero-call analysis fixes the aggressive first checkpoints to `[6,7,7,9,10]` and permits only one fallback, depth 10. A hindsight oracle stops early exactly when the fresh answer already reaches the requested fidelity; otherwise it pays a second full Target call at depth 10.

## Result

| Policy | Anchor successes | Complete | Mean final context/query | Mean Target compute/query |
|---|---:|---:|---:|---:|
| Aggressive fixed | 5538 | 775 | n/a | n/a |
| Depth10 fixed | 6191 | 1123 | 3191.06 | 4019.72 |
| Two-stage oracle | 6345 | 1142 | 2375.55 | 3641.12 |

The two-stage oracle saves **815.51 final-context tokens/query (25.56%)** and **378.60 actual Target prompt+generation tokens/query (9.42%)** relative to depth10 fixed, while using 1.164 calls per attainable request (maximum two). It also preserves early successes that depth10 later loses.

Decision: `GO_C1_PREFLIGHT_NATIVE_CONFIDENCE`. This authorizes only a small traced-path equivalence preflight. It does not authorize a classifier or claim deployability.
