# PROJECT-SYNTHESIS-A0: does a good solution exist?

This audit separates three claims that earlier experiments often mixed:

1. **representation existence**: a short, nested sequence in the frozen
   Target's native symbol space exists;
2. **Target consumption**: frozen Qwen3-8B actually uses that sequence;
3. **automatic learnability**: a deployable compressor can infer it from only
   query and original context.

Only the first claim can be proved without implementing or training a new
method. The second needs a small Target positive control. The third cannot be
proved from an oracle construction and requires held-out learning evidence.

## Constructive existence witness

For every one of the 256 C1 design queries, order the canonical correct answer
strings once. For a query with `n` answers, the shortest false-positive-free
prefix reaching set-F1 `tau` contains

`ceil(tau*n/(2-tau))`

answers. The five such prefixes are nested and consist entirely of Qwen3-8B
vocabulary tokens. This is deliberately a non-deployable gold construction:
it proves existence in the representation space and nothing about inference.

The cumulative five-level code is 193.88 Target positions/query on average,
versus 2,316.96 V8 context tokens, a ratio of **8.37%**. At every anchor, all
256 answer-code prefixes are shorter than the corresponding V8 prefix.
Therefore compactness, nesting, and native-symbol compatibility are not
mathematically inconsistent.

The corresponding exact-source-proof construction costs 2,049.24 positions,
or **88.45%** of V8 cumulatively. It is especially poor at 0.80, where it is
99.84% of V8. Thus simply retaining full source witnesses has almost no margin
for construction cost or errors. The remaining theoretical opportunity lies
in lossy task-relevant native codes, not concatenated source proofs.

## What is and is not proved

The audit disproves the strongest impossibility claim: a compact progressive
Target-native representation exists. It does **not** show that Qwen3-8B will
consume it under the frozen QA protocol, that a compressor can infer it
without effectively solving QA, or that end-to-end compute improves after code
construction. Since the witness contains gold answers, using it as a method or
compressor input would be leakage.

The next admissible experiment is therefore only
`NATIVE-CODE-A0_CONSUMPTION_POSITIVE_CONTROL`: on a small design-exposed
cohort, test whether frozen Qwen3-8B reliably converts oracle native answer
prefixes into the requested answers at all five budgets. No compressor is
trained. Failure closes discrete representation before implementation;
success establishes only Target consumption and authorizes a separate
learnability audit.

Decision: `GO_NATIVE_CODE_CONSUMPTION_POSITIVE_CONTROL`.

No Target call or sealed-set access occurred.

