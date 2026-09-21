# SEM-E0Q constrained proposition linker positive control

SEM-E0Q tested whether frozen Qwen3-14B could act as a narrow offline linker
rather than a fact generator. It saw only the question, question schema, and
numbered source sentences. It could return `ABSTAIN` or exact source quotes for
subject, predicate, objects, constraints, and one frozen link type. Gold answers,
Qwen3-8B outputs, fidelity, and repair/break labels were not loaded.

The 96-query design cohort and prompt hash were frozen before generation.
Generation was greedy at batch size one. This experiment made 96 Teacher calls,
40,369 prompt tokens, and 4,699 generated tokens; mean generation latency was
2.28 seconds/query. It made zero Target calls.

| Result | Count |
|---|---:|
| Outputs passing JSON/exact-quote hard checks | 60/96 |
| Hard-valid `LINK` outputs | 46 |
| Semantically valid linked propositions | 25/46 (54.3%) |

The hard checker verifies provenance and output shape, not entailment. A protocol
bug also allowed a document-title subject with `SAME_CLAUSE` even when the title
was absent from that sentence; strict semantic review rejected those cases.
Failures covered the preregistered mechanisms: subject swap, subject/object role
inversion, wrong predicate, missing constraints, unresolved coreference, answer
type mismatch, and argument mismatch. The known “The Drones” album/performer
subject-swap and Higher-Broughton mismatch were both incorrectly linked.

Decision: `STOP_RELATION_EXTRACTION_ENGINEERING_DO_NOT_OPEN_FRESH_E0C2`.

The 14B constrained positive control is far below the precision required for a
fresh evidence gate. Re-prompting on this exposed cohort would begin an E0Q1/Q2
relation-extraction tuning branch that the protocol explicitly intended to
avoid. E0C2 and Qwen3-8B Target evaluation remain closed. The project should now
reassess whether a task-specific provenance-backed representation can avoid
open-domain proposition extraction, or whether the semantic-evidence branch has
become too distant from the primary progressive-compression objective.

Human IAA was unavailable. Semantic labels are AI-assisted design judgments and
are not presented as independent validation.
