# V0.4 sentence-atomic + frozen semantic top-six

## Decision

V0.4 is complete and independently artifact-valid, but its preregistered
scientific gate is **NO-GO**. V1/QLoRA remains stopped. Retrieval is no longer
the failing component; three downstream rate--fidelity checks fail.

## Completed chain

- 30 fresh examples, 180 sentence-atomic units, and no gold construction flags.
- Frozen candidate SHA-256:
  `0cc341ee79aec571d464d3035c524d129a76a2c4bdb00cac3cb99962c29073a0`.
- Top-six selection retained 65/71 gold facts (91.55%).
- Full selected-context Qwen3-8B: EM 46.67%, mean F1 0.5937, and 14/30
  examples with F1 at least 0.8.
- Frozen Qwen3-14B: 180/180 validated gist/residual packets. Three one-fact
  atomic sentences legitimately have an empty residual because their gist
  already contains every content-bearing word.
- Frozen Qwen3-8B: all `30 * 729 = 21,870` raw and fact-aware exact states.
- Of 65 selected facts, gist supports 18 and gist+residual supports 58; full
  packets add support for 40 facts beyond gist.

## Frozen primary-grid result

The gate has 41 adjacent feasible pairs across 16 examples. Only 2 pairs
strictly increase rate (4.88%, required 20%). Nested reuse is 1/2=50%
(required 80%). Across ten examples feasible at both 0.60 and 0.90, mean fact
recall gain is 0.0583 (required 0.15, or 15 percentage points).

The remaining checks pass: artifact integrity, sample/state counts, at least
30 adjacent pairs, mean normalized structural gap 0.086% (maximum 10%), gap
P90 zero (maximum 20%), no fact-recall declines, 91.55% retrieval coverage,
and label independence.

One primary-grid structural gap remains identifiable. Example
`5a77857155429949eeb29ebf` has a 73-token independent optimum at level 0.60
and a 110-token optimum at 0.70. The best nested chain pays five extra tokens
at 0.60, a 6.85% structural tax. Attainable-breakpoint sensitivity contains
21 strict rate changes and 26 positive gap rows spanning two examples; it does
not replace the preregistered primary grid.

## Interpretation

The semantic selector solved the measured retrieval bottleneck, and residuals
do carry substantial missing fact content. Nevertheless, the fixed primary
thresholds are coarse relative to the discrete target outputs: at levels 0.80,
0.90, and 0.95 the same ten feasible examples all have answer F1 and fact
recall equal to 1.0, so these thresholds share the same optimum. This is a
scientific mismatch between the current fidelity observable and the desired
multi-level frontier, not an implementation defect to tune away after seeing
the locked result.

The defensible next research decision is to redesign and preregister the
fidelity observable/evaluation population, or reconsider whether V1 can learn
anything from this sparse primary signal. The current gate must not be
retroactively changed and does not authorize V1.
