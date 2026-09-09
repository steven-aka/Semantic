# Retrieval R0: bounded label-free multi-hop screen

## Decision

The CPU-only retrieval screen completed, but the locked test did not reach the
pre-registered 80% top-six gold-fact coverage requirement.  The decision is
`STOP_RETHINK_N6`; V0.4 was not constructed and no GPU model was started.

## Frozen design

- Excluded all 130 IDs used by V0--V0.3.
- Deterministically selected 500 development and 200 locked-test examples by
  salted SHA-256 rank.
- Kept sentence-atomic units, six selected units, visible/token-counted titles,
  BM25 `k1=1.2`, and `b=0.75`.
- Compared exactly three development policies:
  1. global sentence BM25;
  2. three paragraph titles with two sentences per title;
  3. a linked title pair with three sentences per title.
- The highest development fact coverage selected the sole policy evaluated on
  the locked test.  Ties were resolved by the predeclared policy order.
- Selector code never reads answers or supporting-fact annotations.  Synthetic
  label-perturbation tests pass for all three policies.

## Results

| Policy | Development fact coverage | Development support-title coverage |
|---|---:|---:|
| Global BM25 | 66.37% | 80.50% |
| Hierarchical 3x2 | 65.14% | 72.30% |
| Linked title pair 2x3 | **75.86%** | 78.00% |

The linked-title-pair policy was selected.  On the 200-example locked test it
retrieved 376/497 gold facts, or **75.65%**, and 78.75% of supporting titles.
This is below the required 80%, so no `data/units/hotpot_v0_4_candidates.jsonl`
file was emitted.

## Interpretation

The linked pair substantially improves sentence-level fact coverage over
global BM25, confirming that multi-hop structure matters, but fixed six-slot
allocation creates a conflict.  Global BM25 covers more titles but often picks
the wrong sentence within a title; pair-based retrieval spends enough sentences
inside a title but sometimes chooses the wrong title pair.  More hand-written
lexical rules would be post-test tuning and are disallowed.

The next scientifically defensible choice is between (a) keeping six units and
using a frozen semantic/cross-encoder evidence ranker, or (b) changing the
controlled exact-search budget and search method.  Either choice is a new
experiment with a fresh locked test split and requires an explicit decision.
