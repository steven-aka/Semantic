# M0 QAMPARI atomic-binary exact-search result

Status: **PASS. The scientific gate is established for V1 experiments.**

Date: 2026-09-10

## Method

Each example contains 12 source-preserving, label-free, independently
revealable packets. A state is a 12-bit vector, so all `2^12 = 4096` states
are evaluated with the frozen Qwen3-8B target. Independent optima are compared
with the globally optimal nested chain computed by Boolean-lattice dynamic
programming. A nested chain is represented exactly by one reveal threshold per
packet.

This replaces the earlier fixed short-block/gist and long-block/residual roles.
It measures the intended decomposition before learning:

```text
independent optimum -> nested optimum -> learned threshold policy
                    G_struct          G_learned
```

## Frozen fresh30 result

The final fresh set was selected from official QAMPARI test eligible positions
30--59 and locked before its exact search, redesign training, calibration, or
policy evaluation. All 122,880 states were present and unique.

| Check | Fresh30 result | Gate |
|---|---:|---:|
| Complete examples | 30 | >=30 |
| Exact states | 122,880 | 4096/example |
| Packet/source/evidence/ceiling integrity | 30/30 | all |
| Strict rate transitions | 117/117 = 100% | >=20% |
| Nested reuse, any optimal tie | 88.89% | >=80% |
| Non-nested independent switches | 15 | >=1 |
| Mean atom-recall gain, 0.60 to 0.90 | 0.4000 | >=0.15 |
| Atom-recall decline fraction | 0 | <=10% |
| Full-state F1 >= 0.8 | 30/30 = 100% | >=90% |
| Any-state F1 >= 0.9 | 30/30 = 100% | >=90% |
| Mean normalized structural gap | 0.002758 | <=0.10 |
| Normalized structural-gap p90 | 0.002482 | <=0.20 |
| Positive structural-gap rows | 16 | diagnostic |

Every conjunctive check passed with `errors=[]`. The earlier zero structural
gap was therefore not preserved by construction: independent optima switch
non-nestedly, and nesting has a small but measurable cost.

Development (`dev10`) and the first heldout (`heldout30`) also passed. The
first heldout was subsequently consumed into the bounded V1 redesign training
set after the primary learned policy failed; it was not reused for the final
policy claim.

## Scientific boundary

M0 establishes that a low-cost nested semantic refinement path exists on the
evidence-certified, target-answerable population. It does not establish that a
learned policy can recover that path. The latter is tested separately in
`results/v1_atomic/RESULTS.md`.

Authoritative artifacts:

- `configs/m0_qampari_atomic_binary_fresh30_gate.json`
- `results/m0_qampari_atomic/fresh30_gate_result.json`
- `results/m0_qampari_atomic/fresh30_gate/independent_frontier.csv`
- `results/m0_qampari_atomic/fresh30_gate/nested_frontier.csv`
- `results/m0_qampari_atomic/fresh30_gate/structural_gap.csv`
- `results/m0_qampari_atomic/fresh30_exact/`

Final decision:

```text
complete=true
errors=[]
scientific_gate_passed=true
decision=GO_V1_ATOMIC_ORDINAL_READINESS_NOT_TRAINING
```
