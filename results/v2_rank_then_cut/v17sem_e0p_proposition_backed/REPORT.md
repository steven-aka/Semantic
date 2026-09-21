# SEM-E0P proposition-backed constructor preflight

E0P tested a proof object with source offsets and an explicit proposition-link
type. It used only the design-exposed D0.5 cohort and made no Teacher/Target
calls.

A stricter review first corrected two E0 labels that omitted explicit Sweden
and summer-2020 constraints, leaving 19/29 valid design emissions. For these 19,
E0P stored the document-title subject span, known-argument spans, enclosing
source sentence and offsets, predicate schema, and link type.

The full enclosing sentence is a conservative safe witness, not the minimal
closed clause. Its Qwen3-8B token lengths were:

| Diagnostic | Count |
|---|---:|
| Minimum / median / maximum | 22 / 54 / 83 |
| `<=16` | 0/19 |
| `<=24` | 1/19 |
| `<=32` | 4/19 |

This does not establish that minimal proposition witnesses exceed these budgets.
It establishes that sentence-level provenance alone is too expensive and that
the missing component is reliable clause/proposition extraction. Offsets prove
where text came from but do not prove subject–predicate–object entailment.

No dependency parser, SRL package, or frozen model for this purpose is available
in the reproducible runtime. Introducing an unconstrained model at this point
would add a new learned component before its scientific role is defined.

Decision: `STOP_BEFORE_FRESH_E0C2_IMPLEMENT_OR_FREEZE_PROPOSITION_LINK_VALIDATOR`.
The next branch must either implement and validate a constrained
proposition-link parser on the exposed regression suite, or define a
provenance-backed structured packet whose relation link is explicitly audited.
Fresh evidence precision and Target utility remain unopened.
