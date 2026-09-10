# Consumed fresh30 ordering/cutoff decomposition

Status: **complete diagnostic; not a new heldout claim**

Date: 2026-09-10

The former fresh30 was already consumed by the failed absolute-threshold V1
evaluation. It is reused here only to diagnose that frozen checkpoint before
defining Rank-then-Cut. No training is performed in this analysis.

## Four-way decomposition

The learned order sorts packets by the old policy's predicted reveal
threshold. An oracle cutoff is the globally optimal nondecreasing sequence of
prefix lengths for that fixed order. `Oracle order + learned count` optimizes a
nested packet order while holding the old policy's raw packet counts fixed; it
does not use an arbitrary within-anchor tie order.

| Ordering | Cutoff/count | Anchor contract | All-anchor trajectories | Mean trajectory rate regret on feasible examples |
|---|---|---:|---:|---:|
| Oracle | Oracle | 100% | 30/30 = 100% | 0 |
| Learned | Oracle | 140/147 = 95.24% | 24/30 = 80% | 0.03859 |
| Oracle | Learned raw count | 79/147 = 53.74% | 5/30 = 16.67% | 0.18435 |
| Learned | Learned raw count | 64/147 = 43.54% | 3/30 = 10% | 0.21266 |

The learned ordering is imperfect, especially at `c=0.95`, but is already much
closer to the nested oracle than the learned cutoff. The old absolute threshold
scale is therefore the dominant failure, while ordering remains a secondary
research target.

Per-level learned-order + oracle-cutoff contract success is 100%, 100%, 100%,
96.67%, and 77.78% for `c=0.60,0.70,0.80,0.90,0.95`. Per-level oracle-order +
learned-count success is only 23.33%, 36.67%, 56.67%, 80%, and 74.07%.

## Near-optimal ambiguity sensitivity

Near-optimality is defined over complete nested chains using cumulative rate.
Slack is normalized by `active anchors * full-state tokens`; a state is a set
member only if it occurs in at least one complete chain within the bound.

| Normalized slack | Packet-anchor ambiguity | Identifiable pairs | Learned pairwise accuracy | Robust terminal never-reveal packets |
|---:|---:|---:|---:|---:|
| 0 | 1.78% | 1602 | 80.90% | 50 |
| 0.005 | 26.57% | 1290 | 84.11% | 43 |
| 0.01 | 39.00% | 1052 | 86.60% | 41 |
| 0.02 | 52.67% | 766 | 89.56% | 30 |

This sensitivity confirms that a single tie-broken oracle chain creates many
unsupported ordering labels. The apparent increase in pairwise accuracy with
slack does not mean ranking improves; uncertain pairs are removed from the
denominator.

At the proposed primary 0.5% slack, the old policy recalls only 53.49% of
robust never-reveal packets. The stricter `robust harmful` subset contains 20
packets, of which only 35% are predicted never-reveal. Harmful and merely
redundant packets must remain distinct labels.

## Decision

```text
ORDERING = PARTIALLY_LEARNED_BUT_NOT_SUFFICIENT_AT_HIGH_FIDELITY
CUTOFF = PRIMARY_FAILURE
SINGLE_CHAIN_SUPERVISION = AMBIGUOUS_UNDER_SMALL_RATE_SLACK
NEXT = FREEZE_RANK_THEN_CUT_AND_TRAIN_RANKING_ONLY_FIRST
```

Authoritative artifacts:

- `results/v1_atomic/fresh30_rank_cut_decomposition.jsonl`
- `results/v1_atomic/fresh30_rank_cut_decomposition_summary.json`
- `src/evaluation/rank_cut_decomposition.py`
- `src/search/rank_then_cut.py`
- `src/search/near_optimal_chain_set.py`
