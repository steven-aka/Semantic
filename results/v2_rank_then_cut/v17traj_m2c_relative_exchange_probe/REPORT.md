# TRAJ-M2C: STAY-anchored relative exchange probe

M2C asks whether the M2B hindsight opportunities can be identified from deployment-visible text. It is a three-fold query-grouped OOF diagnostic on the 128 design-exposed M2B train queries, with no new Target calls. Each action is represented by `(query, S6, promoted rank10 atom, delayed rank7 sentence)`. A fixed DistilBERT classifier predicts `harm / neutral / safe Complete repair`; the action score is `P(safe)-P(harm)`, each query takes its maximum-scoring action, and all unselected queries use STAY. Hyperparameters, 5/10/20% query budgets, and the 10% gate were frozen before fitting. No checkpoint is saved.

The 223 actions contain 19 safe-repair, 70 harm, and 134 neutral labels. OOF safe-action average precision is **0.101** versus prevalence **0.085**; harm AP is **0.329** versus prevalence **0.314**. These are negligible improvements. Selective replay is also negative:

| Intervention budget | Switches | Opportunity queries found (random expectation) | Actual safe repairs | Complete breaks | Complete |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5% | 7 | 1 (0.82) | 1 | 1 | 52/128 |
| 10% | 13 | 1 (1.52) | 1 | 3 | 50/128 |
| 20% | 26 | 2 (3.05) | 2 | 5 | 49/128 |

The frozen 10% gate required at least three safe repairs, at most one Complete break, net Complete gain of at least two, nonincreasing context cost, and repairs in at least two folds. It fails decisively: **`STOP_M2C_RELATIVE_EXCHANGE_PROBE`**. Token use falls because exchanges are budget-neutral or saving, but this is paid for by quality loss; it is not a Pareto improvement.

This result closes the current cheap-text selector for the narrow M2 action. It does not prove that all extractive scheduling is impossible, because the cohort has only 15 positive queries and the atom splitter is imperfect. It does show that more labels or a larger encoder are not justified by an existing OOF signal. The next branch must change the action representation itself—most plausibly a compact semantic evidence unit—and must first establish its costed action-space ceiling without gold-answer leakage before training another controller.

[Protocol](../../../configs/v17traj_m2c_relative_exchange_probe.json), [trainer](../../../src/training/train_v17traj_m2c_relative_exchange_probe.py), [summary](summary.json), [query OOF ranking](oof.jsonl), [action OOF predictions](oof_actions.jsonl).
