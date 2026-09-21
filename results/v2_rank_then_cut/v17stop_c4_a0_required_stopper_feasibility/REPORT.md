# V17-STOP-C4-A0 required-stopper feasibility

This zero-Target-call audit asks whether one final independent stopping judge
has a realistic operating region on the 256-query C1 design cohort. It uses
the canonical prompt order and the frozen aggressive checkpoints
`[6, 7, 7, 9, 10]`, with depth 10 as fallback.

## Deployment-aligned label

Ordinary early-sufficiency TPR/FPR is not the correct abstraction. An early
failure that also fails at depth 10 (FF) should stop: this saves compute
without reducing quality. An early success that rolls back at depth 10 (SF)
should also stop. The only class that must continue is FS: early failure
followed by depth-10 success.

The simulated axes are rescue recall, `P(CONTINUE | FS)`, and the
unnecessary-continue rate, `P(CONTINUE | SS, SF, or FF)`.

## Result

The depth-10 baseline is `245/244/242/220/183`, Complete `183/256`, at
4,185.21 Target compute tokens/query. The decision classes are:

| Level | SS | SF | FS | FF |
|---|---:|---:|---:|---:|
| 0.60 | 218 | 5 | 27 | 6 |
| 0.70 | 218 | 4 | 26 | 8 |
| 0.80 | 186 | 3 | 56 | 11 |
| 0.90 | 176 | 10 | 44 | 26 |

Under the frozen one-percentage-point quality/Complete tolerance and Target
compute cap of 0.98 times depth 10, the first feasible point requires **0.966
rescue recall**, even when the unnecessary-continue rate is zero. At a 1%
unnecessary-continue rate, 0.90 rescue recall gives expected Complete 175.79;
0.95 rescue recall gives 179.30. Both fail the quality gate. Approximately
0.97 recall enters the feasible region.

Compute is not the binding constraint. The feasible unnecessary-continue rate
can extend to about 16.4%, because unnecessary fallback primarily spends
compute. It does not relax the requirement to identify almost every FS state.

## Decision

`STOP_STRONGER_STOPPER_BEFORE_TRAINING`

An auxiliary judge would need near-oracle recall on rare, query-specific
rescue states. Existing output, native-confidence and internal-state probes
already fail well before that standard. The remaining compute margin also
excludes the auxiliary judge's own cost. Training another judge is therefore
not justified.

This closes adaptive stopping under the current text/lossless/frozen-Target
contract. It does not invalidate progressive compression. The supported
current method is V8 with a fixed causal schedule; a new adaptive branch now
requires an explicitly revised contract and separately named study.

No sealed split was read and no Target call was made.
