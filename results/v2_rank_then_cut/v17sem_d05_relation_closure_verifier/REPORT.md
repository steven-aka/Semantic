# SEM-D0.5 relation-closure verifier

SEM-D0.5 tested whether a precision-first relation-closure verifier could make
the SEM-D0 exact-span generator safe enough to justify fresh Qwen3-8B calls.
It made no Teacher or Target calls and did not read sealed evaluation sets.

## Design-set diagnosis

The 56 invalid SEM-D0 emissions were assigned one primary mechanism. The most
common were `NO_RELEVANT_FACT` (15), `MISSING_ARGUMENT` (13),
`MISSING_PREDICATE` (11), and `WRONG_PREDICATE` (8). Truncation contributed to
27/56 errors and 29/56 occurred on multi-constraint questions. Exact-source
provenance therefore did not imply relation closure.

A conservative rule verifier was designed on these 96 exposed examples. It
retained 17 candidates, all 17 of which passed the existing adversarial audit,
covering 73.9% of the valid SEM-D0 emissions. This was a design-set sanity check,
not evidence for the gate.

## Fresh gate

The gate used 96 SHA-ordered canonical training queries absent from every prior
experiment manifest. It generated candidates from every adjacent atom pair in
the V8 rank-10 packet, then applied the frozen verifier. AI-assisted,
Target-outcome-blind review was used because independent human annotation was
unavailable. The reviewer could see the frozen verifier decision, so this is
not verifier-blind annotation and is reported as a limitation.

| Metric | Result | Frozen requirement |
|---|---:|---:|
| Source universes with a legal closed <=16-token extract | 77/96 | diagnostic |
| Lexical generator emissions | 90/96 | diagnostic |
| Verifier emissions | 17/96 | >=20 and >=15% coverage |
| Valid verifier emissions | 16/17 (94.1%) | >=95% |
| Precision Wilson 95% lower bound | 73.0% | >=80% |
| Recall over eligible source universes | 16/77 (20.8%) | >=50% |

The single false emission treated birth in **Higher Broughton** as closing a
query for **Broughton**. More broadly, conservative template support reduced
false emissions but abstained on most compositional and long-tail relations.

## Decision

`STOP_OR_REDESIGN_RELATION_CLOSURE`.

SEM-D1 Target evaluation remains unauthorized. More labels or a looser
threshold would not address the mechanism. The next admissible hypothesis is a
source-backed structured extractor that first identifies the requested
predicate and arguments, then selects exact spans that jointly fill those
slots. It must pass the same fresh precision/recall gate before any Target call.

This result does not establish human-level annotation reliability: human IAA
was unavailable and all annotations are explicitly marked AI-assisted.
