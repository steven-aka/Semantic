# V18-L0B-1 32-slot tiny memorization

L0B-1 tested only whether the frozen V18 architecture could transmit enough
query-specific information through 32 latent slots for a frozen Qwen3-8B to
reproduce its V8 depth-10 behavior on 32 design-exposed queries. It did not
test generalization, halting, or Pareto quality.

The frozen Qwen3-1.7B encoded each query plus original context once. Only one
cross-attention resampler and 2048-to-4096 projection were trained. Teacher
answers were loss targets and never compressor inputs. Qwen3-8B had zero
trainable parameters. The pre-registered run used 320 updates and evaluated
both teacher-forced NLL and cached free generation.

## Result

- initial NLL: 4.1591
- final NLL: 2.1727
- relative reduction: 47.76% (gate: at least 70%)
- free-generation teacher answer-set F1: 0.0313 (gate: at least 0.90)
- learned slot effective rank: 32/32
- slot norm range: 0.442--0.523
- NaN/Inf: none
- trainable Target parameters: 0

The loss falls, so gradients and the embedding interface work. The full slot
rank and finite norms rule out a trivial constant-slot collapse. Nevertheless,
free generations mostly contain unrelated or repeated answer items and do not
reproduce the teacher behavior. The gap is too large to interpret as a minor
threshold miss.

Decision: `STOP_LATENT_CHANNEL`.

Per the frozen gate, do not run nested-budget L0B-2, extend steps, add slots,
change encoder/resampler capacity, or start query-held-out L0C. This result is
specific to the one frozen V18 architecture; it does not prove that every
possible latent compressor is impossible. It does show that opening an
architecture-search program would exceed the deliberately narrow final rescue
authorized after V17.

No sealed set was read. No additional Target-label generation was performed;
the experiment used cached design-exposed depth-10 teacher behavior.

