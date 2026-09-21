# Trained frozen-Target baselines

Both baselines keep Qwen3-8B frozen and train only the context compressor or
projector. Training and model selection use train-side query splits; the common
Screen-64 set is used once for end-to-end comparison. The lineage holdout 581,
internal300, development, and confirmation sets remain sealed.

## QAMPARI-domain abstractive RECOMP

- Compressor: T5-large initialized from the released NQ RECOMP checkpoint.
- Supervision: Qwen3-14B query-focused summaries; the frozen filter requires at
  least 0.60 answer-atom recall and no more than 256 T5 tokens.
- Data: 1906/2032 training and 523/550 validation summaries passed.
- Selected checkpoint: epoch 3; validation loss 0.632149.
- Checkpoint SHA-256: `451d622ad65983a173aa6324b2aef78b624c089fd45dd985b87d6218ed84cb71`.

Screen-64 with frozen Qwen3-8B:

| Metric | Full context | Abstractive RECOMP |
|---|---:|---:|
| Mean F1 | 0.9495 | 0.7733 |
| Success @ 0.60 | 63 | 54 |
| Success @ 0.70 | 61 | 51 |
| Success @ 0.80 | 60 | 43 |
| Success @ 0.90 | 53 | 27 |
| Success @ 0.95 | 39 | 14 |
| Mean context tokens | about 743 (implied by ratio) | 77.16 |
| Actual keep rate | 1.0 | 0.1039 |

At 0.90 there are 5 repairs and 31 breaks. The compressor is highly compact but
does not preserve enough answer coverage to meet the project's quality contract.

## xRAG-style Qwen3 projector

- Qualification: Qwen3 port of the xRAG mechanism; the official implementation
  supports Mistral/Mixtral and therefore this is not an official checkpoint.
- Frozen components: Qwen3-8B Target and QAMPARI-adapted 768-d retriever.
- Trainable component: 19,931,136-parameter MLP projector producing one latent
  context embedding.
- Selected checkpoint: epoch 3; validation answer-token NLL 2.256737.
- Target gradients: zero; Target fingerprint unchanged after training.
- Peak allocated CUDA memory in preflight: 19,446,410,240 bytes.
- Checkpoint SHA-256: `e02d5015a78130d6485a5fb7fde770fbcf98518b64bd123a6b0d5680a281fd34`.

Screen-64 with frozen Qwen3-8B:

| Metric | Full context | xRAG-style one token |
|---|---:|---:|
| Mean F1 | 0.9495 | 0.0339 |
| Success @ 0.60 | 63 | 0 |
| Success @ 0.70 | 61 | 0 |
| Success @ 0.80 | 60 | 0 |
| Success @ 0.90 | 53 | 0 |
| Success @ 0.95 | 39 | 0 |
| Latent context tokens | n/a | 1 |

Lower validation NLL did not translate into usable list generation. Under this
data scale and projector-only contract, one continuous token is too restrictive
for QAMPARI's multi-answer evidence.

## Method implications

The two results bracket a useful design range. A readable 77-token abstractive
summary retains substantial signal but loses too many answers; a single latent
token loses almost all task utility. A plausible contribution to the proposed
method is therefore a **small multi-unit bottleneck with explicit multi-answer
coverage supervision**, while retaining progressive reveal and a frozen Target.
This is recorded as a design hypothesis only; no proposed-method retraining was
started from these baseline results.
