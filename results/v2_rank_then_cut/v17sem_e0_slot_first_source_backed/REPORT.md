# SEM-E0 slot-first source-backed construction: design audit

SEM-E0 reversed the SEM-D0 ordering: parse the requested relation first, require
the known arguments and predicate cues, and only then minimize an exact-source
span under 16 Qwen3-8B tokens. No Teacher/Target calls or sealed sets were used.

The question-only parser represented all 96 design-exposed D0.5 questions in
18 template families. This establishes syntactic coverage only; it does not
prove that the normalized relation is semantically correct.

The first slot-first constructor emitted 40/96 candidates. Requiring an answer
type/domain cue reduced this to 29/96. A stricter second pass of the
AI-assisted design review judged 19/29 (65.5%) to close every requested
relation and domain constraint. Ten failures remained:

- missing country/domain or temporal constraints (Belize, Philippines,
  Missouri, Sweden, summer 2020);
- `Higher Broughton` treated as exact `Broughton`;
- absent or mismatched predicates (`Mush Records`, director vs music director);
- missing answer-type/linkage evidence; and
- a subject-linkage error where the source says an album won an award but the
  serialization `The Drones ; won ...` makes the band appear to be the winner.

The final case is decisive: token-set coverage is not proposition closure, and
joining a document title to an arbitrary exact fragment can change who bears a
relation even though every output token is source-backed.

Decision: `NOT_READY_FOR_FRESH_E0C`. The closure-first research direction is
retained, but this implementation is stopped before a fresh cohort. The next
constructor must preserve a source proposition link between answer-bearing
subject, predicate, known argument, and constraints. Every eligible annotation
must store exact quotes, character offsets, covered slots, and tokenizer-checked
length. The earlier D0.5 `77/96` Boolean feasibility estimate is not treated as
a validated exact-witness ceiling because it lacks those witness records.

Human IAA was unavailable; all semantic judgments are marked AI-assisted.
