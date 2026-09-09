# Fidelity signal bottleneck: root-cause resolution and final M0 decision

Status: development experiment complete and independently verified; scientific
gate is **NO-GO**; V1/QLoRA remains stopped.

Date: 2026-09-08

## 中文结论

研究目标没有改变：学习一个查询条件化、由 fidelity 控制、信息只增不减
的加法语义表示，并测量它相对各档独立最优表示的 successive-refinement
结构代价。表示仍只有 `{0: omit, 1: gist, 2: gist+residual}` 三态，rate
仍是目标模型 tokenizer 的真实 token 数，六单元仍精确枚举 `3^6=729`
个状态。

根因已经被分层定位。HotpotQA 通常只有两个关键事实，binary fact recall
把五个 fidelity 档压缩成极少数硬断点；连续 YES/NO 置信度和答案 NLL
又不能可靠代表“信息是否被保留”，因此被实验否决，而不是继续叠加机制。
随后改用官方 QAMPARI 开发集的多答案任务，以 alias-aware list F1 作为
硬 fidelity，并把每题规范为十个答案原子、五个相关双证据单元和一个
干扰单元。

生成式 free-text residual 暴露出 identity closure 与截断问题。最终采用
确定性的 lossless additive partition：每个单元的两段原始证据被无损、
互斥且穷尽地分为 gist 与 residual；短段作为 gist，另一段作为 residual。
packetizer 不读取 question 或 gold annotation，不再依赖生成模型，也不需
继续增加修复规则。

这项改造解决了结构信号瓶颈：104/104 个相邻可行档位都发生严格 rate
增长，出现 16 次非嵌套独立最优切换，nested reuse 为 87.5%，平均
0.60→0.90 answer-atom recall 增益为 0.40，结构差可以稳定测量。唯一未
通过的是冻结的 target–benchmark answerability 门槛：只有
25/30=83.33% 的样本在
所有 729 个状态中能达到 F1≥0.9，低于预注册的 90%。因此联合门槛为
NO-GO，实验按约定停止在 V1 之前。

## 1. Unchanged research contract

- one shared query-conditioned, fidelity-indexed representation;
- monotone additive states `{0, 1, 2}`;
- `state=1` reveals gist and `state=2` reveals gist plus residual;
- rate is retained Qwen3-8B tokenizer tokens;
- six units and exactly 729 states per example;
- fixed grid `{0.60, 0.70, 0.80, 0.90, 0.95}`;
- compare independently optimal states with the best nested chain;
- do not enter V1 unless every frozen scientific gate passes.

## 2. Root-cause sequence

1. **Hotpot granularity:** most examples expose only two gold facts, so hard
   fact recall is commonly limited to `{0, 0.5, 1}`. Five requested fidelity
   anchors consequently collapse to the same constraint.
2. **Confidence proxy rejected:** binary YES/NO margins can be continuous but
   do not make semantic retention continuous. The mechanism was removed.
3. **Contract NLL rejected:** in a real target-model smoke, complete evidence
   could worsen answer NLL and incorrect context could aggregate better. NLL
   is therefore not a valid fidelity contract for this experiment.
4. **Task granularity corrected:** official QAMPARI development data provides
   multiple answer atoms. Alias-aware list F1 is used directly as the hard
   observable; atom recall is retained as a secondary audit.
5. **Free-text closure rejected:** generated residuals sometimes renamed or
   truncated answer-bearing strings. Local repair rules did not guarantee
   closure and were not extended indefinitely.
6. **Representation root fix:** lossless source partition makes gist+residual
   exactly reconstruct each unit while retaining a real additive rate path.

## 3. Frozen development evidence

- Official raw development file: `data/raw/qampari/dev_data.jsonl`.
- Full-context baseline: 60 examples; mean full F1 `0.7720`, mean empty F1
  `0.0339`, mean gain `0.7381`; 37 examples passed both eligibility checks.
- Frozen population: the first 30 jointly eligible examples in pre-existing
  order, each with ten answer atoms and six units.
- Controlled examples SHA-256:
  `728a91d73d80f4466597fd836af330c50721af8286d10a47976bd4bc0b0eb95f`.
- Annotation SHA-256:
  `6001e763b742ec9537c21f9532e8b04f45c0c56712d802ddb758cdcd35089d4f`.
- Selection-manifest SHA-256:
  `c3b90d3e78d1c628ef879ff7add79d94cc4693c6bf500c1807f47febe39fe0bf`.
- Lossless packet tree: 180 validated packets; SHA-256
  `4f983035fbd56bcb674b29cb03422b1a99a866d248ef74267afa4cf0586f4fbd`.
- Exact-search tree: 30 × 729 = 21,870 states; SHA-256
  `c2009deac8324b1aae5ceb98f48e304153cbab14c2767c0a6ab0e80dd71a84f2`.
- Frozen gate config: `configs/m0_qampari_lossless_dev_gate.json`.
- Independent result: `results/m0_qampari/lossless_dev_gate_result.json`.

The controlled evidence construction uses official answer proof documents to
create a diagnostic benchmark. It is not an unbiased retrieval benchmark.
Within that frozen benchmark, however, the lossless packet partition itself
uses only source blocks and their target-tokenizer lengths; it reads neither
the question nor answer labels.

## 4. Final conjunctive gate

| Check | Result | Threshold | Pass |
|---|---:|---:|:---:|
| Complete examples | 30 | ≥30 | yes |
| Exact states | 21,870 | 729/example | yes |
| Lossless packets | 180 | all | yes |
| Strict rate increases | 104/104 = 100% | ≥20% | yes |
| Non-nested independent switches | 16 | ≥1 | yes |
| Nested reuse, any optimal tie | 87.5% | ≥80% | yes |
| Mean atom-recall gain, 0.60→0.90 | 0.400 | ≥0.15 | yes |
| Atom-recall declines | 0 | 0 | yes |
| Full-state F1≥0.8 | 27/30 = 90% | ≥90% | yes |
| Any-state F1≥0.9 | 25/30 = 83.33% | ≥90% | **no** |
| Example-weighted normalized gap | 0.00534 | ≤0.10 | yes |
| Normalized structural-gap p90 | 0.00848 | ≤0.20 | yes |

Additional diagnostics: 104 state switches, 15 positive structural-gap rows,
and all-ties nested reuse `82.69%`. Independent verification reports
`complete=true` and `errors=[]`.

## 5. Scientific interpretation

### Resolved

- The earlier zero structural gap was a measurement/granularity artifact.
- The representation now produces progressive, strictly increasing rate.
- Independent optima switch often enough to identify structural cost.
- High-fidelity states largely reuse lower-fidelity content.
- Gist plus residual has deterministic source closure.
- Numeric equivalence, title identity, cache integrity, hashes, exact state
  counts and independent verification are covered by tests and audits.

### Still unresolved

- Five of 30 development questions cannot reach list F1≥0.9 under any of the
  729 controlled states. A post-gate read-only audit found mixed causes: all
  five best predictions contain eight matched atoms; some omissions are true
  target enumeration failures, while three examples also expose incomplete
  query-condition evidence, a noisy boundary answer, or a Wiki display-name
  mismatch. The result therefore identifies target–benchmark compatibility,
  not target capacity alone and not a missing compression transition.
- No learned compressor has been trained to imitate exact optima.
- Transfer to natural documents, other datasets and other target models is
  untested.
- A fresh locked test has not been run because the development gate failed.
- A learned abstractive residual with the same closure guarantee remains an
  open later research problem.

### Rejected actions

- add packet states merely to create more operating points;
- revive confidence margins or answer NLL as semantic fidelity;
- add per-error rewriting and validation exceptions indefinitely;
- weaken official hard F1 or the frozen 90% answerability threshold after seeing
  results;
- inspect QAMPARI test target outputs after a failed development gate;
- start V1/QLoRA despite a failed conjunctive prerequisite.

## 6. Final stop rule

The structural bottleneck is solved, but the full frozen gate is not. The
authoritative decision is:

```text
complete = true
errors = []
scientific_gate_passed = false
decision = NO_GO_STOP_BEFORE_V1
```

No additional M0.x repair is justified on this development population. A
future continuation must be a newly motivated study of target–benchmark
answerability and evidence-contract design, with new preregistration and data
separation; it cannot reinterpret this result as a pass.
