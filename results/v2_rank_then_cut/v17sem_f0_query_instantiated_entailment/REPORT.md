# SEM-F0 query-instantiated entailment final rescue

## Question

Can deterministic query instantiation remove the proposition-construction
failure seen in SEM-E0Q? For each of 96 design-exposed queries, the candidate
entity is the exact rank-10 document title. A question-only schema creates a
complete hypothesis. Frozen Qwen3-14B then judges that hypothesis against the
future packet and V8 S6 independently as `ENTAILED`, `NOT_ENTAILED`, or
`ABSTAIN`. It may quote source text but may not generate a new fact.

This is the pre-registered terminal rescue for automatic semantic evidence
construction. It uses no Qwen3-8B Target outcomes, gold answers, or sealed
evaluation queries.

## Execution

- 96 queries and 192 paired Teacher tasks.
- Greedy Qwen3-14B, batch size 1, maximum context 4096.
- Longest preflight prompt was about 900 tokens.
- 181/192 outputs passed JSON, decision-enum, and exact-quote checks.
- Eleven failed mechanically: five truncated/no-JSON outputs and six quotes
  that were not exact source substrings.

Among hard-valid outputs, 61 future packets were judged entailed and no S6
prefix was judged entailed. Thus the automatic policy would emit 61 novel
candidates on this exposed cohort.

## Semantic regression result

The high emission count does not pass the gate. The frozen regression review
passed 12/17 cases and failed IDs 11, 22, 31, 40, and 46. Three failures are
clear false-positive relation/type errors:

- ID 11 attributes an award won by an album to the band *The Drones*.
- ID 22 treats “evolved from Birmingham-based groups” as entailing that *The
  Move* was formed in Birmingham.
- ID 31 treats a German brewery owner who later studied archaeology as an
  explicitly supported member of the queried explorer class.

Two additional failures are false negatives on frozen positive controls (IDs
40 and 46). A conservative review therefore finds at least three invalid
emissions among the 61 hard-valid emissions: 58/61 = 95.1% precision, with a
95% Wilson interval of 86.5%–98.3%. This is not an independent precision
estimate: the cohort is design-exposed and the review is AI-assisted with no
human inter-annotator agreement. The three-error count is intentionally a
clear-error lower bound, not a claim that every other emission is correct.

## Decision

`STOP_AUTOMATIC_SEMANTIC_EVIDENCE_CONSTRUCTION`.

The test is materially better than SEM-E0Q, so query instantiation did remove
many answer-role and cross-span failures. It did not remove the remaining
semantic entailment boundary: subject attribution, query-class membership,
and close but non-equivalent relations still cross the gate. The terminal
contract forbids prompt revision, a larger Teacher, or another relation
extractor on this cohort. Fresh F0B and Qwen3-8B Target testing remain closed.

The scientifically defensible asset is the negative result: source-backed
compact evidence cannot yet be constructed automatically at the precision
needed to protect a strong V8 baseline. Future work should return to the
primary compression objective with either a fixed, already validated baseline
or a genuinely different representation whose action-space value and
deployment cost are established before training.
