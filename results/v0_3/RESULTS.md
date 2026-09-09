# V0.3 sentence-atomic + label-free BM25

## Decision

V0.3 is complete and artifact-valid, but the preregistered scientific gate is
**NO-GO**.  V1/QLoRA must remain stopped.  This is not an implementation
failure: all 30 examples, 180 teacher packets, and `30 * 3^6 = 21,870` exact
states completed and independently verified.  Three of ten preregistered gate
checks failed.

## Frozen protocol

- The 30 examples were selected from HotpotQA validation IDs not used by
  V0/V0.1/V0.2, using the lowest SHA-256 ranks under the fixed salt
  `v0_3_label_free_sentence_atomic_20260907`.
- Each original sentence is one unit.  The title is model-visible and included
  in the token count.  There is no overlap and no unit exceeds 256 target
  tokenizer tokens (observed range 10--115).
- Per-example BM25 (`k1=1.2`, `b=0.75`) ranks `title + sentence` against the
  question.  The top six units are selected and rendered in source order.
- Construction does not use the answer or supporting-fact annotations.  A
  label erasure/permutation reconstruction produced identical unit signatures.
- Gold supporting facts are used only after candidate construction.  Missing
  top-six facts remain missing and contribute zero fact recall.
- Teacher, target, three representation states, hard-min fidelity, exact state
  count, and primary fidelity grid are unchanged from V0.2.

The frozen thresholds and decision rule are in `configs/v0_3_gate.json`; the
candidate IDs and data hashes are in `results/v0_3/data_manifest.json`.

## Results

- Label-free top-six selection retained 51/72 gold facts (70.83%), below the
  preregistered 80% threshold.
- The frozen Qwen3-14B produced 180/180 valid packets.  Of the 51 selected gold
  facts, Qwen3-8B judged 11 supported by gist and 46 supported by
  gist+residual.  The 21 unselected facts were scored as absent.
- Full selected-context Qwen3-8B F1 was at least 0.8 on 14/30 examples; EM was
  46.67% and mean F1 was 0.4955.
- The primary grid contains 36 adjacent feasible pairs across 11 examples.
  Five pairs have strict state/rate changes (13.89%), below the preregistered
  20% threshold.
- Nested reuse is 4/5 = 80% under both the any-tie and all-ties definitions,
  exactly meeting its threshold.
- The example-weighted mean normalized structural gap is 0.394%, and the P90
  is zero, passing the 10% and 20% limits.  There is one positive primary-grid
  gap.
- Across the nine examples jointly feasible at 0.60 and 0.90, mean fact recall
  increases by 0.1111, below the required 0.15.  No adjacent feasible
  transition decreases fact recall.
- The attainable-breakpoint sensitivity grid has 15 state/rate switches, two
  nonnested switches, and four positive-gap rows.  It does not replace the
  primary grid.

The single positive primary case is `5a79305755429907847277dd`.  At fidelity
0.60 its independent optimum is `[0,2,0,2,0,2]` at 189 tokens with achieved
fidelity 2/3.  At 0.70 the optimum is `[0,2,2,1,0,2]` at 230 tokens with
fidelity 1.0.  The change upgrades unit 2 while downgrading unit 3, so the best
nested chain must already use the 230-token state at 0.60.  The structural tax
is 41 tokens, or 21.69%.

## Interpretation

Sentence-atomic, label-free units preserve the important V0.2 finding that the
structural gap can be activated without gold-aware boundaries.  The main
failure is upstream: a single-hop lexical top-six selector misses 29.17% of
gold facts in a multi-hop dataset.  Packet content loss and low selected-context
answerability further reduce the number of active primary-grid transitions.

The bounded conclusion is therefore: do not change the three-state
representation and do not start V1.  V0.3 is the one preregistered label-free
BM25 run and must not be retuned after observing its result.  Any future work
on a multi-hop label-free candidate selector is a new, separately approved
experiment rather than a hidden V0.3 retry.
