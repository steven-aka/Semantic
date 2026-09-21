# LATENT-L0A legality and accounting audit

## Decision

`STOP_LATENT_UNDER_CURRENT_LOSSLESS_TEXT_API_CONTRACT`.

The current project defines its method as deterministic, lossless,
source-preserving, text-rendered, and add-only. This is stated in the README,
the V2 hypothesis, packet validation, exact-search code, and all canonical V8
deployment paths. The canonical Target runner renders ordinary chat text and
token IDs. It has no deployment contract for external `inputs_embeds` or
prefix KV state.

A learned latent sequence could preserve two properties in principle: Qwen3-8B
parameters could remain frozen, and a single maximum latent sequence could be
revealed through nested prefixes. It cannot preserve the current lossless text
claim. The vectors are not a recoverable partition of source text, and using
them requires a white-box interface that is outside the current deployment
contract. Counting latent positions as text tokens would also be invalid;
compressor compute, Target positions, latency, and memory would require a new
multi-axis evaluation.

Because legality is conjunctive, no embedding-equivalence test, compressor
training, Teacher cache, or Target evaluation is warranted. LATENT-L0B remains
closed.

This does not assert that latent compression is technically impossible. It
could be studied as a separately named, lossy, white-box representation method
after explicitly changing the research question and deployment constraints.
Such a study would not be a continuation of the present lossless method and
could not be compared using text-token compression claims alone.

With both PACKET-H0 and LATENT-L0 stopped at their preflights, V8 remains the
current defensible deployable method. The remaining scientific result is the
measured oracle--deployment gap and the non-monotone, query-specific response
of the frozen Target, rather than an unvalidated new compressor.
