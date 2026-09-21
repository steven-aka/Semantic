# FRONTIER-R2: xRAG Frozen-Target Eligibility Audit

## Question

Can the official pretrained xRAG checkpoint be evaluated as another baseline
under the current contract: original QAMPARI context, no training, and the same
frozen Qwen3-8B Target?

## Evidence

- Official repository commit: `121fa4180a8c1fa0ec1af5901d879452e5c9ce89`.
- Official `xrag-7b` backbone: `mistralai/Mistral-7B-Instruct-v0.2`.
- Official retriever: `Salesforce/SFR-Embedding-Mistral`.
- The released `Hannibal046/xrag-7b` checkpoint contains approximately 14.55
  GB of Mistral-bound model weights.
- The training code's frozen-decoder mode updates the projector only, confirming
  that the projector is learned for the decoder representation space rather
  than being a model-independent context format.

## Decision

`BLOCK_XRAG_QWEN3_BASELINE_REQUIRES_BRIDGE_TRAINING`

The official checkpoint cannot be connected to Qwen3-8B without learning a new
bridge. Running it unchanged would replace the Target with Mistral and would no
longer be a comparable baseline for this project. Learning a Qwen3 bridge would
be a new method-training experiment, which the current baseline-only phase
explicitly forbids.

No Target call, model download, training, or sealed-set read occurred. This is
an eligibility result, not a negative performance result for xRAG.

## Deferred mechanism note

xRAG remains relevant if training is later authorized: it is a clean test of
whether a learned frozen-decoder-specific carrier is more consumable than the
external latent channel already rejected by V18. That possibility is recorded
only; it does not trigger training.
