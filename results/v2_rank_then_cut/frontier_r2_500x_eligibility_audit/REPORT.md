# FRONTIER-R2: 500xCompressor Eligibility Audit

## Question

Can 500xCompressor be evaluated now as a no-training baseline with the same
frozen Qwen3-8B Target?

## Evidence

- Official repository commit: `ff454a1669e8616698ff1c775aa6ab1db718bea8`.
- The official README states that uploaded datasets and models are not open to
  the public.
- Released code targets `meta-llama/Meta-Llama-3-8B-Instruct` and loads learned
  LoRA/compression parameters into that backbone.
- The method preserves an original decoder path, but its learned encoder/LoRA,
  compressed tokens, and KV carrier are decoder-backbone specific.

## Decision

`BLOCK_500X_BASELINE_NO_PUBLIC_CHECKPOINT_AND_QWEN3_PORT_REQUIRES_TRAINING`

There is no public official checkpoint to reproduce. A Qwen3-8B version would
require pretraining/fine-tuning a new compressor, which is outside the current
baseline-only phase. Substituting random LoRA weights or another latent model
would not be a 500xCompressor reproduction.

No model download, Target call, training, or sealed-set read occurred. This is
an asset/interface eligibility result, not a negative performance result.

## Deferred mechanism note

The paper's evidence that KV carriers outperform embedding carriers at extreme
ratios remains relevant to the project's failed V18 embedding channel. A
controlled Qwen3 KV-versus-embedding experiment may be useful if training is
later authorized, but it is recorded only and does not trigger retraining.
