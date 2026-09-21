# V18-L0A progressive latent contract and interface preflight

V17 is frozen after C4 showed that adaptive stopping on the existing text
trajectory needs approximately 96.6% recall on rescue-required FS states. V18
is a separately named successor that preserves the final objective and frozen
Qwen3-8B, but explicitly retires the lossless, source-preserving, text-only,
black-box representation contract.

L0 is a representation-feasibility experiment, not an oracle ceiling. Fewer
latent positions do not establish lower total cost; Target and compressor
positions, latency, FLOPs, VRAM and cache memory must be reported separately.
SF reduction alone is also gameable by making every short prefix fail, so the
gate requires at least 50% lower SF at matched per-anchor success.

The one frozen candidate uses a frozen Qwen3-1.7B encoder once per query, one
small trainable cross-attention resampler and a projection into the Qwen3-8B
4096-dimensional embedding space. Nested budgets are `[4,8,16,24,32]`.
Architecture/budget sweeps and adaptive halting are excluded from L0.

The static screen found the required `inputs_embeds` and cache APIs. Existing
C1 traces give 4,185.21 depth10 Target token-positions/query over five calls.
Projected latent Target positions are 943.71; a transparent 1.7B/8B
parameter-position proxy adds 165.84 for the once-per-query encoder, giving a
screening ratio of 0.265. This is not a deployment cost claim.

The exact Qwen3-8B checkpoint was then tested in bf16 on a synthetic prompt:

- token IDs versus exact `inputs_embeds`: maximum logit difference 0;
- cached token-ID versus cached embedding suffix: difference 0, top-1 100%;
- latent gradient is finite (norm 20.54), with every Target gradient `None`;
- embedding-path latency was 0.967 times the token-ID path in this short test.

One-shot versus cached token-ID logits differed by up to 0.594 in bf16. Since
the two cached interfaces are identical, this is a one-shot/cached numerical
path effect, not a latent-interface failure. Future comparisons must use the
same cached execution path.

Decision: `GO_V18_L0B_TINY_MEMORIZATION_PREFLIGHT`. This authorizes only a
small overfit/plumbing test, not full training, adaptive halting, sealed-set
access, or a compression-improvement claim.
