# V0.2 fact-atomic structural-gap stress pilot

V0.2 keeps the original HotpotQA questions and answers, frozen Qwen3-14B
packet teacher, frozen Qwen3-8B target, additive `gist + residual`
representation, three states `{0,1,2}`, and
`fidelity=min(answer_f1, gold_fact_recall)`. It changes only the V0 oracle
unit granularity: each gold supporting sentence is one unit, and non-support
HotpotQA material fills the controlled example to six units.

## Data and inference

- 100 original candidates contained 33 questions with 3--5 gold facts.
- All 33 controlled examples passed the unit audit: 6 units per question,
  198 units total, 5--211 target-tokenizer tokens, no empty or over-limit unit.
- Frozen full-context Qwen3-8B retained 17/33 examples at F1 >= 0.8:
  13 with three facts, 3 with four facts, and 1 with five facts.
- Frozen Qwen3-14B produced 102/102 validated packets.
- Frozen Qwen3-8B judged 56 gold facts in gist and full packet states. Gist
  explicitly supported 9/56; gist+residual supported 51/56.
- Exact search completed all `17 * 3^6 = 12,393` states and rescored them with
  the content-aware fact judgments.

## Primary preregistered grid

At fidelity levels `{0.60,0.70,0.80,0.90,0.95}` there are 44 adjacent
feasible pairs, 12 state/rate/fact-recall increases, one nonnested independent
switch, and one positive structural-gap row. The diagnostic assessment is
`structural_gap_identifiable`.

The positive case is HotpotQA example `5a8e1027554299653c1aa15f`:

- At level 0.60, the unique independent optimum is state
  `[0,1,2,0,0,0]`, 34 tokens, achieved fidelity 2/3.
- At levels 0.70 and 0.80, the unique independent optimum is state
  `[0,1,1,0,2,0]`, 51 tokens, achieved fidelity 6/7.
- The transition is nonnested because unit 2 is downgraded while unit 4 is
  upgraded. The best nested chain must instead use `[0,1,1,0,1,0]` at level
  0.60, costing 36 tokens.
- Structural tax at level 0.60 is therefore 2 tokens, or 5.88%. Across the 15
  examples feasible at this level, the mean normalized tax is 0.392%.

## All attainable fidelity breakpoints

After merging floating-point-equivalent breakpoints, the diagnostic grid has
213 adjacent feasible pairs, 31 strict state/rate/fact-recall increases, five
nonnested switches, and 23 positive structural-gap rows across five examples.
The largest observed tax is 47 tokens (22.38%). This grid is a sensitivity
analysis and does not replace the primary grid.

## Interpretation and boundary

The earlier zero gap was caused by a coarse two-support-unit pilot that did not
activate the nesting constraint. V0.2 provides a real positive control under
the unchanged objective and shows that structural tax is measurable when gold
facts are independently addressable. Because fact-atomic unit construction
uses gold support labels, this is an oracle structural stress pilot, not an
unbiased end-to-end benchmark. V1 remains stopped pending the human V0 gate.

## Final chain verification

`scripts/09_run_v0_2.sh all` was rerun after the artifacts were complete. It
reconstructed the deterministic data, validated and reused every GPU cache
without loading vLLM, recomputed both analyses, and passed the independent
end-to-end verifier. The final manifest is `results/v0_2/verification.json`:
`complete=true`, `errors=[]`, 33 candidates, 17 eligible examples, 102 packets,
56 gold facts, and 12,393 exact states. All 31 tests and dependency checks pass.
