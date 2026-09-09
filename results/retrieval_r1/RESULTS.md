# Retrieval R1: frozen semantic top-six selector

## Decision

Retrieval R1 passed both preregistered gates and emitted the 30 frozen V0.4
candidates. On a fresh 200-example locked test, the frozen Qwen3-14B selector
retained 430/481 gold supporting facts (89.40%) against an 80% threshold. It
used the deterministic BM25 fallback on 3/200 examples (1.50%) against a 2%
maximum.

## Protocol integrity

- All 830 IDs consumed by V0--V0.3 and Retrieval R0 were excluded.
- A salted hash fixed 20 parser-smoke and 200 locked-test IDs before model
  output. The longest formatted prompt was 3,491 tokens.
- The selector process loaded only `id`, `question`, and `context`; answers and
  supporting-fact labels were first loaded after all 220 selector records were
  cached and the complete cache tree was hashed.
- The model was local frozen Qwen3-14B BF16 with greedy decoding. It selected
  exactly six original sentences and received no answer or support labels.
- Support-title coverage was 388/400 (97.00%), and the selected units used 3.9
  unique titles on average.

The protocol is frozen in `configs/retrieval_r1.json`. Split, cache, metric,
and decision hashes are recorded in `prepare_manifest.json`,
`selection_manifest.json`, and `manifest.json` in this directory.

## Interpretation

This resolves the upstream six-slot retrieval failure seen in V0.3 and
Retrieval R0 without changing sentence atomicity, the state budget, fidelity,
or gold-label policy. Passing retrieval is necessary but not sufficient for
the full V0 scientific gate; that independent test is V0.4.
