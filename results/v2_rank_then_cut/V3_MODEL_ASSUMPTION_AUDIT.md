# V3 model-assumption audit (2026-09-15)

Status: completed post-hoc failure analysis on **consumed** train2000,
development300, and rank-confirm210. No new heldout or model-selection claim.
The locked calibration300 and final-test300 were not read.

## Definitions and frozen scope

The Target is unchanged: frozen Qwen3-8B fidelity on each complete 4096-state
binary lattice. A harmful packet-anchor group is one excluded packet at one
fidelity anchor that makes at least one admissible near-optimal state fall below
the anchor when added. At the V3-frozen normalized slack 0.005, the audit
compares its add-one fidelity deltas across all admissible states at that **same
anchor** where the packet is absent. Strict sign flip requires both a negative
and a positive delta; a zero delta alone is not a strict sign flip. Contract
crossing flip means addition breaks the anchor in some such states but not
others. These are different statistics. Slack uses extra cumulative tokens
divided by active-anchor count times full-state tokens, not a percentage of
the optimum cost.

Raw local precedence edges come from critical-delete versus harmful-add
witnesses. Stable edges are the subset identifiable across the complete
near-optimal nested-chain set. A raw local cycle therefore describes
state/anchor-dependent *labels*, not proof that no feasible static order
exists. The stable graph is globally chain-consistent and acyclic in all
audited examples.

| Consumed role | Strict sign-flip groups | Contract-crossing flip groups | Raw local cycle examples | Stable graph cycle examples |
|---|---:|---:|---:|---:|
| Train2000 | 2941/12083 = 24.34% | 7437/12083 = 61.55% | 460/2000 = 23.00% | 0/2000 |
| Development300 | 391/1734 = 22.55% | 1088/1734 = 62.75% | 68/300 = 22.67% | 0/300 |
| Confirm210 | 291/1179 = 24.68% | 762/1179 = 64.63% | 53/210 = 25.24% | 0/210 |

| Consumed role | Boundary pair accuracy | All labeled boundaries strictly separated | Oracle-cutoff complete trajectory |
|---|---:|---:|---:|
| Train2000 | 94.66% | 77.09% of 1772 supervised | 1841/2000 = 92.05% |
| Development300 | 92.19% | 66.92% of 266 supervised | 270/300 = 90.00% |
| Confirm210 | 92.64% | 70.33% of 182 supervised | 187/210 = 89.05% |

The train-to-development complete-boundary gap is 10.17 percentage points;
train-to-confirm is 6.76 points. There is no development-to-confirm collapse,
and train is far from perfect. V3 therefore combines a modest generalization
gap with an optimization/supervision failure already visible on train. The
late train-loss improvement alongside development-loss deterioration is
consistent with overfitting, but it is not the sole scientific bottleneck.

## Near-optimal ambiguity sensitivity

| Normalized slack | Train ambiguous packet-anchor fraction | Development | Confirm | Train mean stable boundary edges | Development | Confirm |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.04% | 2.07% | 2.12% | 14.47 | 13.87 | 13.20 |
| 0.005 (V3 frozen) | 26.10% | 26.43% | 26.69% | 19.02 | 18.99 | 18.38 |
| 0.01 | 38.68% | 39.60% | 39.95% | 20.66 | 20.74 | 20.26 |
| 0.05 | 70.50% | 71.30% | 70.75% | 12.13 | 11.30 | 12.87 |

Five-percent slack admits many more alternative states and actually *reduces*
stable boundary labels; it should be treated as sensitivity analysis, not a
retroactive replacement of the frozen 0.5% label rule. This audit checks
set-identifiable direction, not a uniform probability over all chains;
the latter would require explicit chain-counting and an arbitrary chain prior.

## Decision

1. Keep `STOP_V3_RANKING`. The rank-confirm risk gate failed at 0.90 and 0.95,
   so no cutoff training, calibration, or final-test evaluation is authorized.
2. Do not claim that sign flips or raw cycles refute static total ordering.
   M0 exact nested trajectories and V3 oracle-label projection show that many
   contract-feasible static orders exist. Conversely, acyclic stable labels do
   not prove that a learned sparse-edge decoder reaches those prefixes.
3. Do not select a V4 pairwise or sequential architecture from these post-hoc
   rates alone. A single controlled V4 formulation should be frozen only
   after defining its deployable decoding and **oracle-cutoff contract** gate;
   pair accuracy, HSR, and graph acyclicity are diagnostics, not substitutes.
4. Before V4 training, freeze a new target-blind confirmation population
   outside the consumed train/development/confirm and locked
   calibration/final-test roles. Precompute the Bonferroni one-sided
   Clopper--Pearson required success counts at each active anchor from that
   population size. Reusing confirm210 for model selection would invalidate
   the scientific gate.

Artifacts: `v3_model_assumption_audit_train_oracle.json`,
`v3_model_assumption_audit_dev_confirm.json`, selected V3 train/development
oracle-cutoff evaluation summaries, and `v3_rank_confirm_decision.json` in
this directory. The audit implementation and reproducible runner are
`src/evaluation/v3_model_assumption_audit.py` and
`scripts/44_run_v3_model_assumption_audit.sh`.
