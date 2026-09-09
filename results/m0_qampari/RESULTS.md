# M0 QAMPARI lossless development result

The end-to-end development chain is complete and independently reproducible,
but its frozen conjunctive scientific gate is **NO-GO**. V1 was not started.

The deterministic lossless additive representation resolves the earlier flat
frontier: all 104 adjacent feasible pairs strictly increase rate, there are 16
non-nested independent-optimum switches, nested reuse is 87.5%, and mean
answer-atom recall gain from 0.60 to 0.90 is 0.40. The normalized structural
gap is small but nonzero and identifiable (example-weighted mean 0.00534; p90
0.00848; 15 positive rows).

The sole failed numerical gate is target–benchmark answerability. Only 25/30
examples have any state with alias-aware QAMPARI list F1 at least 0.9, below
the frozen 90% requirement. The threshold was not changed after observing the
run.

A read-only audit of the five failures shows that every best prediction has
eight matched answers (`F1=0.8889`), but the cause is mixed rather than a clean
model-capacity result:

| Example | Audit finding |
|---|---|
| `38__wikitables_composition__dev` | Includes boundary/noisy `Titanic` for “after 1997” and a Wiki-markup Harry Potter label. |
| `474__wikitables_simple__dev` | Selected proof sentences for two omitted bridges do not state the required >600 m condition. |
| `69__wikitables_composition__dev` | Both omitted atoms are explicitly supported; this is an eight-of-ten target enumeration failure. |
| `511__wikitables_simple__dev` | Both omitted typhoons are supported; this is another eight-of-ten target enumeration failure. |
| `795__wikidata_simple__dev` | One omitted leader's selected proof does not identify that answer; the other is a target omission. |

Thus this gate cannot isolate target capacity from evidence-contract noise. It
does establish that the frozen population is not sufficiently answerable for
V1, while leaving the structural signal result intact.

Authoritative artifacts:

- `configs/m0_qampari_lossless_dev_gate.json`
- `data/units/qampari_m0_dev30_sentence.jsonl`
- `data/packets_qampari_m0_dev_sentence_lossless/`
- `results/m0_qampari/dev_sentence_lossless_exact_search/`
- `results/m0_qampari/lossless_dev_gate_result.json`
- `results/m0_qampari/lossless_exact_frontier.csv`
- `results/m0_qampari/lossless_nested_frontier.csv`
- `results/m0_qampari/lossless_structural_gap.csv`

Final decision:

```text
complete=true
errors=[]
scientific_gate_passed=false
decision=NO_GO_STOP_BEFORE_V1
```

No QAMPARI test target inference and no V1/QLoRA training was performed.
