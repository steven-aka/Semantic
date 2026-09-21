# TRAJ-M1 low cumulative cost borrow

This is a prospective **train-side research test**, not a deployment or sealed-set result. The intervention is the same exact rank-10 title-plus-sentence atom selected in M0B: reveal it at depth 7, retain it through depth 9, and restore the original full rank-10 packet at depth 10. The five read depths remain `[6,7,7,9,10]`. Only the action's measured five-level cumulative context slack changes: borrow when it is at most **72 tokens**, otherwise use V8 unchanged. The cap and decision gate were frozen before the fresh Target calls in [the config](../../../configs/v17traj_m1_low_cost_borrow.json).

The prospective population consists of all 116 remaining M0B-eligible train queries after excluding R1, OBS-A1, FRAG-B0, and M0C. Only 47/116 satisfy the new cap. We used 558 fresh Qwen3-8B calls for paired baseline/action evaluation; action and baseline share the same depth-6 and depth-10 contexts. No internal, development, confirmation, or lineage-clean 581 queries were read. This population is smaller than the suggested 256 because only 116 minimally exposed eligible queries remained; it has limited power and is not an independent final confirmation.

| Metric | V8 | M1 uniform policy | Paired change |
| --- | ---: | ---: | ---: |
| 0.60 success | 108/116 | 108/116 | 0 repair / 0 break |
| 0.70 success | 104/116 | 103/116 | 1 repair / 2 break |
| 0.80 success | 86/116 | 91/116 | 6 repair / 1 break |
| 0.90 success | 79/116 | 81/116 | 6 repair / 4 break |
| 0.95 success | 77/116 | 77/116 | unchanged by the action |
| Complete | 62/116 | 66/116 | 6 repair / 2 break |

Uniform action costs **+18.08 five-level cumulative context tokens/query** and **+29.10 Target prompt-plus-output tokens/query**. The latter is measured, not treated as interchangeable with context tokens. The paired Complete difference is +4/116; a query bootstrap (10,000 draws, seed 20260921) gives a 95% interval of roughly **−1 to +10 queries**, so the observed improvement is uncertain.

The hindsight no-anchor-break Complete oracle chooses six actions with slacks `[33,33,42,42,42,45]`, or **39.5 extra cumulative context tokens per repaired query**. This is a ceiling that uses forbidden Target outcomes, not an available controller. The pre-registered gate required at least eight such repairs, at most 50 extra tokens per repair, and no uniform Complete regression. It passes the cost and non-regression clauses but fails the opportunity-count clause: **`STOP_M1_EXTRACTIVE_BORROW_GATE`**. We do not change the eight-repair requirement after seeing six.

Interpretation: the cheaper action preserves some 0.80/0.90 and Complete benefit, but its safe opportunity set is too sparse in this untouched cohort to justify selector training under the frozen gate. The result does not prove that any form of extractive borrow is impossible, nor that generated semantic evidence compression is necessary. It only closes this rank-10, depth-7, single-atom, ≤72-cumulative-token contract. A different action mechanism would require a new independently frozen hypothesis and sufficient untouched train-side data. The full 0.95 problem remains separate. The renderer preserves the evidence set add-only; it re-renders in source order, so contexts are not guaranteed to be literal string prefixes.

Artifacts: [manifest](manifest.json), [summary](summary.json), [per-query outcomes](per_query.jsonl), [raw paired Target outputs](per_call.jsonl), [runner](../../../src/evaluation/v17traj_m1_low_cost_borrow.py).
