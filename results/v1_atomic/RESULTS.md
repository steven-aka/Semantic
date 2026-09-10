# V1 atomic reveal-threshold policy result

Status: **NO-GO for the current learned policy. M0 remains valid.**

Date: 2026-09-10

## Frozen method

The policy uses Qwen3-1.7B with QLoRA and one shared scalar head over each
packet's local hidden state plus the query-conditioned global hidden state.
It predicts one threshold per lossless packet. At fidelity control `c`, packet
`i` is revealed when `c >= tau_i`, which guarantees representation-level
nestedness. Training uses masked BCE over the complete fidelity trajectory,
not SmoothL1 regression on a single threshold target.

The original train40 policy failed on its locked heldout30: active-contract
success was 50%, all-active-trajectory success was 0/30, and no feasible policy
gap could be reported. A single bounded redesign was declared before further
training: add those consumed 30 examples to the 40-example training pool, keep
the architecture and loss unchanged, use a disjoint calibration16 set, and
evaluate once on a newly locked fresh30 set.

The resulting train/calibration/fresh sizes are 70/16/30 and every pairwise
example-ID overlap is zero.

## Training

The frozen train70 run used 56 training and 14 internal-validation examples
for 10 epochs. Epoch 7 was selected by minimum validation BCE:

| Metric | Result |
|---|---:|
| Best validation BCE | 0.567224 |
| Best validation ordinal accuracy | 0.787990 |
| Best epoch | 7/10 |
| Exact validation trajectory fraction | 0 |
| Elapsed time | 822.9 s |

This improves optimization over the primary train40 checkpoint (best
validation BCE 0.932148), but does not by itself establish fidelity control.

## Calibration16

The predeclared monotone calibration selects the smallest control value in
0.05 increments that attains at least 90% empirical contract success.

| Target `c` | Selected control | Calibration success | Meets 90% |
|---:|---:|---:|:---:|
| 0.60 | 0.85 | 15/16 = 93.75% | yes |
| 0.70 | 0.90 | 16/16 = 100% | yes |
| 0.80 | 1.00 | 15/16 = 93.75% | yes |
| 0.90 | 1.00 | 13/16 = 81.25% | **no** |
| 0.95 | 1.00 | 10/13 = 76.92% | **no** |

The calibration is explicitly recorded as `calibration_valid=false`; falling
back to control 1.0 is not counted as success.

## One-shot fresh30 evaluation

The invalid frozen mapping was nevertheless evaluated once for diagnosis. It
was not tuned or rerun from the fresh result.

| Target `c` | Active examples | Contract success | Independent rate | Nested-optimal rate | Learned rate |
|---:|---:|---:|---:|---:|---:|
| 0.60 | 30 | 26/30 = 86.67% | 0.3184 | 0.3203 | 0.6054 |
| 0.70 | 30 | 29/30 = 96.67% | 0.3935 | 0.3955 | 0.7364 |
| 0.80 | 30 | 30/30 = 100% | 0.4835 | 0.4853 | 1.0000 |
| 0.90 | 30 | 28/30 = 93.33% | 0.6852 | 0.6864 | 1.0000 |
| 0.95 | 27 | 14/27 = 51.85% | 0.8526 | 0.8526 | 1.0000 |

Aggregate results:

```text
representation_nested_fraction = 1.0
ordinal_accuracy = 0.778912
exact_trajectory_fraction = 0
active_contract_success_fraction = 0.863946
all_active_contracts_success_fraction = 0.5
mean_policy_gap_normalized_feasible = 0.353904
mean_total_gap_normalized_feasible = 0.355872
gap_decomposition_max_residual = 5.55e-17
calibration_valid = false
```

The learned policy therefore fails both contract control and rate efficiency.
Infeasible predictions are counted as violations, never as negative gaps.

## Root scientific diagnosis

The failure is not an M0 ceiling failure. On fresh30, every example has some
state with F1 >= 0.9, and 27/30 have some state with F1 >= 0.95. The target is
not monotone under adding evidence: the full state reaches F1 >= 0.9 for only
28/30 and F1 >= 0.95 for only 14/30. Thus mapping high controls to 1.0 reveals
all packets and destroys much of the exact subset advantage.

The threshold representation can express the oracle nested chain on the
frozen fidelity grid, as established by construction and tests. The observed
failure is the learned policy's inability to identify the high-fidelity subset
and generalize its thresholds from 70 examples. Calibration cannot repair
wrong packet ordering; it only moves a global operating point and degenerates
to full context at the upper anchors.

Per the bounded-redesign rule, no additional loss term, decoder loop, or
post-fresh threshold tuning is added. Further work requires a new scientific
hypothesis and a newly frozen protocol, not V1.x patch accumulation.

Authoritative artifacts:

- `configs/v1_atomic_ordinal_redesign.yaml`
- `results/v1_atomic/redesign_train70/training_metadata.json`
- `results/v1_atomic/redesign_train70/history.jsonl`
- `results/v1_atomic/calibration16_mapping.json`
- `results/v1_atomic/fresh30_policy_calibrated.jsonl`
- `results/v1_atomic/fresh30_policy_calibrated_summary.json`
- `data/threshold_labels/qampari_atomic_fresh30_ordinal.jsonl`

Final decision:

```text
M0_ATOMIC_GATE=PASS
V1_ATOMIC_LEARNED_POLICY=NO_GO
STOP_CURRENT_METHOD_WITHOUT_ADDITIONAL_PATCHES
```
