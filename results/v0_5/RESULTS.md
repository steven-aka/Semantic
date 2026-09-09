# V0.5 plan-aligned answerable-population test

## Decision

V0.5 is complete and artifact-valid, but the unchanged preregistered scientific
gate is **NO-GO**. Per the frozen decision rule, V0 revision stops here and
V1/QLoRA remains unauthorized.

## Why this was the final bounded repair

The implementation plan recommends full-context F1 at least 0.8 (or correct
exact match) for compressor supervision. V0.4 had only 14/30 such examples.
V0.5 therefore changed only the evaluation population: from the 170 unused
Retrieval R1 locked-test positions, frozen Qwen3-8B found 85 answerable
examples, and the first 30 in the already-frozen order were selected. No exact
state result was available during selection.

Sentence boundaries, six units, the three `{0,1,2}` states, Qwen3-14B teacher,
Qwen3-8B target/judge, hard-min fidelity, primary grid, and all ten scientific
thresholds remained unchanged.

## Completed artifacts

- 170/170 answerability baseline rows; 85 pass F1 >= 0.8.
- Selected 30: EM 93.33%, mean full-context F1 0.9886, minimum F1 0.8.
- Candidate SHA-256:
  `888afbb28754a28be037ca5f3148f5f345e86ca40d34bd363f920fbe58a392d6`.
- 180/180 validated packets and all `30 * 729 = 21,870` exact states.
- 67/72 selected gold facts = 93.06% top-six coverage.
- Gist supports 13/67 selected facts; gist+residual supports 60/67, so residual
  adds support for 47 facts.
- Independent verification: `complete=true`, `errors=[]`.

## Gate result

Eight of ten checks pass. The answerable population increases primary-grid
adjacent feasible pairs from V0.4's 41 to 82 and raises nested reuse to 5/5 =
100%. Structural gap is zero throughout the primary grid, fact recall never
declines, and retrieval, integrity, sample count, and gap bounds pass.

Two checks still fail:

- Strict rate transitions: 5/82 = **6.10%**, required at least 20%.
- Mean fact-recall gain from 0.60 to 0.90: **0.0833**, required at least 0.15.

The plateau remains visible: levels 0.70 and 0.80 have the same 21 feasible
examples and the same mean optimum rate; levels 0.90 and 0.95 have the same 20
examples and the same mean rate. Attainable-breakpoint sensitivity is richer
(28 state switches, 6 nonnested switches, and 20 positive gap rows), but it is
not the preregistered primary analysis.

## Engineering incidents

Three teacher normalization cases were rejected by the strict packet validator:
a redundant residual, cardinal word-to-digit conversion, and ordinal
grade-range conversion. Deterministic final repairs accept only literal
source-preserving reversals; unseen numbers and new content remain errors.
Targeted tests cover both accepted and rejected cases. Interrupted validator
failures left vLLM children alive; only the exact project-owned failed children
were terminated, and atomic per-example caches resumed without loss.

## Scientific conclusion

Retrieval inadequacy and low full-context answerability have both been ruled
out as explanations for the missing five-level tradeoff. The remaining failure
is the method: this three-state packet representation evaluated by
`min(answer F1, fact recall)` does not provide a sufficiently dense response on
the fixed primary grid. Another V0 patch would be post-result iteration rather
than a bounded diagnosis. A new research phase now requires an explicit change
to the representation or fidelity observable and a new preregistration; it
must not be labeled as a passed V0 or used to silently authorize V1.
