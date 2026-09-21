# FRAG-B0: natural-query title-only reveal pilot

Before any new Target calls, the protocol selected the SHA256-first 128 P0 train queries outside the R1 train512 frame, with an available rank-10 document title. It did not inspect their P0 success outcomes. At the fixed V8 depth-9 stop, each query was rerun in one paired Qwen3-8B batch as (a) ordinary V8 and (b) V8 plus the rank-10 title only. This is an add-only reveal; the canonical depth-10 packet can still be restored. No model was trained and no sealed query set was read.

| Endpoint on 128 natural train-side queries | V8 | Title-only |
|---|---:|---:|
| 0.90 successes | 100 | 101 |
| Paired 0.90 repairs / breaks | — | 6 / 5 |
| Mean extra depth-9 context tokens | 0 | 8.45 |
| Mean extra Target prompt+generated tokens | 0 | 10.78 |

The paired-query bootstrap 95% descriptive interval for net success count is **−5 to +8**; the mean continuous-F1 change is +0.00905 with interval **−0.00080 to +0.01896**. Twenty-nine queries gained F1, ten lost F1, and 89 were unchanged. This is a weak, uncertain quality shift for a universal extra-cost action, not a demonstrated final five-anchor Pareto improvement. Only depth-9/0.90 was paired fresh; other anchors and Complete were not recomputed or claimed.

The selected FRAG-A0 cases had suggested that a title alone could repair some failures cheaply. FRAG-B0 shows why those outcome-enriched cases cannot justify default title reveal: ordinary queries include almost as many breaks as repairs. Since the constructed source pool puts a gold answer in nearly every rank-10 title, these gains also risk measuring an answer cue rather than evidence relation support. A hypothetical title gate remains unproven; the current data do not justify fitting another gate on this exposed cohort or expanding labels for it.

**Decision:** `STOP_UNIFORM_TITLE_ONLY_REVEAL`; retain the V8 trajectory and fixed schedule as the deployable comparison. The combined title+sentence component interaction remains a mechanistic observation, not a learned controller. The action-boundary repair may be adopted as a data-quality change after a separate fixed-protocol evaluation, but it touched only one decisive query in the present pilot and is not the dominant observed bottleneck.

The config, frozen query manifest, per-call outputs, paired per-query table, and `summary.json` make this run reviewable. It used 256 Target calls, totaling 170,079 prompt and 13,877 generated tokens, on the current Qwen3-8B contract.
