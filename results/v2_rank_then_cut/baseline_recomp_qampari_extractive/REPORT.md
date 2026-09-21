# QAMPARI-supervised RECOMP-style extractive baseline

## Contract

- Frozen Target: Qwen3-8B; no Target parameter was loaded during compressor
  training and no Target parameter is trainable.
- Compressor initialization: official public NQ RECOMP extractive checkpoint.
- Training: 1,967 train queries and 35,361 sentence pairs after removing all
  Screen-64 query IDs.
- Model selection: disjoint 550-query inner validation set with 9,837 pairs.
- Supervision: train-only gold answer-alias coverage at sentence level.
- Qualification: in-domain **RECOMP-style adaptation**, not an official
  QAMPARI checkpoint or exact reproduction of an author-released configuration.
- Lineage holdout 581, internal300, development, and confirmation were not read.

## Training result

| Epoch | Train loss | Validation pair accuracy | Mean margin |
|---:|---:|---:|---:|
| 1 | 0.01389 | 0.9677 | 0.5511 |
| 2 | 0.00370 | 0.9670 | 0.6366 |
| **3** | **0.00154** | **0.9693** | **0.6339** |

Epoch 3 is selected by the frozen rule: pair accuracy first, then margin.

## Frozen-Target Screen-64 result

The fresh complete-context control is mean F1 0.9495 and 53/64 at 0.90.

| Requested keep | Actual keep | Mean tokens | Mean F1 | 0.90 success | Repair / break |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.2485 | 200.38 | 0.5308 | 5/64 | 1 / 49 |
| 0.50 | 0.4972 | 400.97 | 0.7239 | 11/64 | 0 / 42 |
| 0.75 | 0.7454 | 600.94 | 0.8516 | 33/64 | 2 / 22 |
| 0.90 | 0.8939 | 721.36 | 0.9224 | 47/64 | 4 / 10 |
| 0.95 | 0.9399 | 758.50 | 0.9287 | 50/64 | 4 / 7 |

At requested 0.50, in-domain training improves substantially over the official
NQ zero-shot checkpoint (mean F1 0.7239 versus 0.5374; 0.90 success 11/64
versus 1/64). It still does not preserve the high-fidelity contract. The mild
0.95 point saves about 6% of context but loses 0.0208 mean F1 and three net
0.90 successes.

## Interpretation

The result confirms that much of the earlier RECOMP failure was cross-domain
ranking mismatch. It also shows that high sentence-level pair accuracy is not
sufficient for distributed answer-set preservation: the final Target remains
sensitive to which supporting sentences are omitted. This baseline is retained
as a trained comparison, with no sealed-set escalation.

The local checkpoint is stored at
`checkpoints/baselines/recomp_qampari_extractive`; its hash is recorded in the
artifact manifest. Generated training data is retained locally and can be
recreated by the committed builder.
