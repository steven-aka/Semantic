# 可用于论文的实验结果汇总

更新日期：2026-09-21  
项目：面向冻结 Qwen3-8B 的 query-conditioned progressive context compression

## 1. 证据范围与论文声明边界

本文件只汇总已经有可复核实验产物支持的结论，并区分四类统计角色：

- **正式训练/验证结果**：协议、模型、数据角色和停止规则在运行前冻结。
- **设计侧或 query-grouped OOF 结果**：可用于方法设计和机制分析，不能称为最终独立验证。
- **oracle ceiling / positive control**：只证明机会或上界存在，不能称为可部署性能。
- **工程与因果审计**：用于排除实现错误、定位失败机制。

目前没有打开最终 sealed confirmation。因此论文现阶段可以报告：V8 是当前合同下最可靠的自动 progressive text policy；大量后续实验系统定位了它的 stopping、表示和 Target compatibility 边界。**不能声称最终系统已在独立测试集上显著优于所有基线。**

## 2. 固定研究设置

- 数据任务：QAMPARI，多答案列表问答。
- Target：冻结 Qwen3-8B，greedy decoding，thinking disabled。
- 原始 V17 表示：12 个 source-preserving、lossless 文本 packet。
- Progressive 约束：add-only nested reveal。
- Fidelity anchors：0.60、0.70、0.80、0.90、0.95。
- 主要质量指标：各 anchor success、Complete trajectory。
- 成本指标：实际 context tokens、累计五档 context、Target prompt/output compute；新增模块另计 latency/FLOPs/VRAM。
- V8 controller：Qwen3-4B LoRA + history-aware sequential policy，beam width 8。

## 3. 当前最强自动文本方法：V8

### 3.1 为什么采用 sequential policy

在 consumed development300 上，105,602 个可由多种历史到达的 selected masks 中，43,930 个（41.60%）会因先前最高 fidelity 状态不同而具有不同的最优下一动作集合；292/300 个样本至少包含一个这种状态。这证明仅根据无序 selected set 的静态 scorer 会面对冲突标签，history-conditioned sequential policy 有直接实验依据。

### 3.2 V8 正式 one-run 结果

V8 单次预注册训练完成 750 optimizer steps，无 OOM 或训练错误，总训练时间 8,418.74 秒。固定 checkpoint 结果为：

| Step | 0.90 success | Complete | Feasible normalized regret | Action accuracy |
|---:|---:|---:|---:|---:|
| 250 | **277/300** | **273/300** | 0.02793 | 0.82802 |
| 500 | 274/300 | 273/300 | 0.02360 | 0.86575 |
| 750 | 273/300 | 272/300 | **0.02287** | **0.86873** |

预注册规则选择 step 250。它通过 Complete ≥0.90 和 regret ≤0.03，但 0.90 success 未达到要求的 282/300，因此正式决定是 `STOP_V8_ONE_RUN`，没有打开 fresh confirmation。

论文可报告的机制结论：action accuracy 和 validation loss 随训练改善，而最终 0.90 success 从 277 降至 273，表明 surrogate optimization 与最终 trajectory contract 存在错位，而不是简单的优化器未收敛。

### 3.3 训练规模并非主要解法

V7 将训练覆盖从 2,000 增至 3,163 个 attainable queries，安全监督边从 38,045 增至 59,562（约 +58%），仍只得到 277/300 的 0.90 success 和 273/300 Complete；相对 V6 是一题 repair、一题 break。单纯扩大相同监督形式的数据没有解决主瓶颈。

## 4. Fresh canonical V8 trajectory 基线

CANON-P0 在 1,421 个 train-side queries 上重新生成了 17,052 个 Qwen3-8B prefix states。固定 schedule 结果：

| Schedule | 0.90 | 0.95 / eligible | Complete | Mean normalized cumulative context |
|---|---:|---:|---:|---:|
| `[10,10,10,10,10]` | 1223/1421 | 973/1131 | **1123/1421** | 0.8201 |
| `[9,9,9,10,10]` | 1223/1421 | 973/1131 | 1084/1421 | 0.7363 |
| `[8,8,9,9,10]` | 1050/1421 | 973/1131 | 942/1421 | **0.6628** |

质量跃升集中在 depth 6、7、9、10，而非平滑增长。该结果支持使用 level-specific schedule，同时揭示了 fixed schedule 的明显质量–token trade-off。

## 5. Ordering 与局部 trajectory modification

### 5.1 Adjacent-swap oracle 与学习结果

在 `[6,7,7,9,10]` 下，V8 为 0.90 `1050/1421`、Complete `775/1421`、累计 context 2183.62。no-break/no-extra-token adjacent-swap oracle达到 0.90 `1067/1421`、Complete `797/1421`、context 2180.59。可部署空间很小：仅 +17 个 0.90、+22 Complete、−3.03 token/query。

对应的 query-grouped OOF learned editor 得到 0.90 `1055/1421`、Complete `781/1421`，但 context 增至 2193.17；产生 26 repair / 21 break。正式结论为 `STOP_PROG_A1_NO_OOF_PARETO`。

### 5.2 跨档 early evidence borrow

在 fresh paired 128-query M0C 中：

| Policy | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 | Complete | Cumulative context |
|---|---:|---:|---:|---:|---:|---:|---:|
| V8 | 116 | 108 | 84 | 93 | 89 | 68 | 2291.14 |
| Early rank-10 atom | 116 | 112 | **100** | **101** | 89 | **79** | 2367.52 |

该动作产生 19 Complete repairs、8 breaks，净增 11，但增加 76.38 cumulative context tokens/query，未形成严格 Pareto improvement。它证明未来细粒度 evidence 提前出现可同时改善 0.80、0.90 和 Complete，但原文搬运的信息密度不足。

### 5.3 Budget-neutral exchange

M2B 在 128 queries、223 actions 上测试 promote-one/delay-one exchange。固定动作节省 36.35 cumulative tokens/query，但 Complete 从 52 降至 44（11 repairs、19 breaks）。hindsight no-break oracle 可修复 15 个 Complete cases，说明动作空间有机会；query-grouped M2C probe 的 safe-action AP 仅 0.101（prevalence 0.085），10% intervention budget 只找到 1 repair、造成 3 breaks，Complete 52→50。机会存在，但部署可见信号不能可靠兑现。

## 6. Candidate coverage 与 cutoff 是两个独立瓶颈

### 6.1 Candidate-pool ceiling

在 606-query design holdout 上，冻结 V13 proposal pool 的 oracle 0.90 coverage 随 K 增长：

| Pool | Oracle coverage |
|---|---:|
| V8 | 567/606 |
| Top-4 | 582/606 |
| Top-10 | 599/606 |
| Top-16 | 602/606 |
| Top-20 | 605/606 |
| Exact lattice | 606/606 |

这是 oracle-pool ceiling，不是 deployable selector result。它证明 rank 5–20 中存在额外成功路径，但没有证明 label-free selector 能可靠找到它们。

### 6.2 Learned cutoff failure

H0 cutoff 在 606-query train-only holdout 上得到 0.90 `510/606`、Complete `360/606`，而相同固定 orders 的 oracle cutoff为 `567/606` 和 `552/606`。在 design-exposed internal300 上，V14+H0 为 0.90 `243/300`、Complete `165/300`，相同 orders 的 oracle cutoff为 `280/300`、`277/300`。

CUT-A0 将 57 个可恢复的 0.90 failures 分为：35 early stops、17 late stops with fidelity rollback、5 previous-level constraint failures。简单 threshold scan 最多只到 `516/606`，不能解决非单调 stopping。

## 7. Adaptive stopping 的上界、可观测性与计算成本

### 7.1 Two-stage clairvoyant ceiling

在 1,421 条 fresh canonical chains 上，“aggressive checkpoint；不足时 fallback depth10”的 clairvoyant oracle相对 fixed depth10：

- final context：2375.55 vs 3191.06，节省 **815.51/query（25.56%）**；
- Target compute：3641.12 vs 4019.72，节省 **378.60/query（9.42%）**；
- Complete：1142 vs 1123；
- anchor-success sum：6345 vs 6191。

这证明 stopping 有真实经济上界，但不提供部署规则。

### 7.2 Native confidence 与 internal state 均不足

256-query query-grouped OOF 中，native-confidence 高阈值点近似保持质量：`245/245/243/219/183`、Complete 182，对比 depth10 `245/244/242/220/183`、Complete 183；但 Target compute 从 4185.21 墧至 5725.41/query。Output-only、native-only、combined probes 均未同时通过质量和计算 gate。

随后收集的 Qwen3-8B final-layer H1/H2/H3 internal states同样没有任何 probe 通过五档 + Complete 的 1pp gate；近似质量点仍比 depth10 更贵。该结果排除了“简单读取 frozen Target hidden state即可解决 stopping”的假设。

### 7.3 Prefix cacheability 与质量冲突

理想 100% evidence-prefix cache reuse 的 ceiling只能将 compute 从 4185.21 降至 4034.50（−150.71，**3.60%**）。实际构造的 reveal-order prefix serialization虽通过 256/256 token-prefix与 source preservation 检查，但 paired depth10 quality从 canonical `123/122/121/108/88` 变为 `121/120/118/109/83`；0.60、0.70、0.80、0.95 均超过 1pp 退化容差。故 cache-compatible order 本身改变 Target 行为。

### 7.4 Stopper 必须接近 oracle

C4 将真实 stopping action重新定义为：只有 FS（early failure、depth10 success）必须 CONTINUE；SS、SF、FF均应 STOP。零调用 operating-region audit显示，即使 unnecessary fallback 为 0，仍需约 **96.6% rescue recall** 才能同时保持五档质量/Complete 在 depth10 的 1pp 内并节省至少 2% compute。1% unnecessary fallback 下：

- 90% rescue recall：expected Complete 175.79/256；
- 95% rescue recall：179.30/256；
- baseline：183/256。

因此 stopping 的主瓶颈是稀有 future-rescue state 的近乎完美识别，而非 classifier 容量或阈值微调。

## 8. Boundary representation 与泛化

D1A 修复了 attainable 0.90 被 progress mask错误抑制的问题，internal300 上 0.90 从 279→280、Complete 274→275，其他 anchors不变；这是一项真实但仅一题的 scoring-semantics repair。

D1B 对 14,087 个 strict 0.90 boundary states、117,153 action pairs进行 frozen representation audit：

| Split | Linear directional accuracy |
|---|---:|
| State-grouped | 0.826 |
| Signature-held-out | 0.820 |
| Query/example-held-out | **0.766** |

非线性 probe 将局部准确率升至 0.883，却使 query-held-out 降至 0.732。D2B rank-8 adapter 三折 macro-state为 0.749/0.721/0.763，均低于对应 baseline；mean 0.744。结论是表示包含局部信号，但稳定的 cross-query decision direction 不成立，增加局部容量会加剧过拟合。

## 9. Semantic/extractive compact evidence 的边界

- 人工 two-atom closure 在 96 条中有 77 条可构造合法闭合证据，证明短 evidence 并非完全不存在。
- 自动 window-first constructor只能正确兑现约 16/77，主要错误与 16-token 截断和关系闭合失败有关。
- constrained proposition linker 即使使用 Qwen3-14B 仍出现 subject/object/predicate attribution errors；最终 SEM-F0 hard-valid policy发出 61 条，保守 review 至多 58/61 = 95.1% precision，Wilson 95% CI 86.5%–98.3%，未达到保护 V8 所需的高置信边界。
- PACKET-H0A 检查 2,032 queries、24,384 packets，确认每个 V8 packet已经是一整个 source block；不存在可无损恢复的隐藏句子/表格行结构。结构性 repacketization并非漏掉的简单修复。

这些结果支持统一解释：source-preserving text 与 Target compatibility 强，但压缩和单调控制有限；自由 semantic representation更紧凑，但自动 faithfulness 不足。

## 10. Latent 与 Target-native code 的终止结果

### 10.1 Continuous latent

V18-L0A 证明 `inputs_embeds`、cached inference和冻结梯度接口正确。32-query、32-slot tiny memorization中：

- NLL：4.1591→2.1727（−47.76%，未达 70% gate）；
- free-generation teacher answer-set F1：0.0313（未达 0.90）；
- slot rank：32/32；无 NaN/Inf；Target trainable parameters：0。

因此失败不是梯度断路或 slot collapse，而是该固定外部 latent channel不能控制 frozen Target 的自由生成。

### 10.2 Constructive native-code existence

使用 gold answers 构造的嵌套 Qwen3-8B vocabulary code在 256 queries 上累计仅 193.88 positions，是 V8 2316.96 的 **8.37%**。这证明 compactness、nesting和 native-symbol compatibility不构成数学矛盾。相同 answers 的 exact source proofs需要 2049.24 positions，即 V8 的 **88.45%**；大压缩率来自答案 sufficient statistic，而非更高效的 source transcription。

### 10.3 Native-code consumption positive control

32 queries × 5 budgets × correct/shuffled，共 320 fresh calls：

- code recovery F1：0.8867；
- nested retention：0.8427；
- correct minus shuffled gold-F1：0.7367；
- direct parse(code) gold-F1：0.8375；
- Target(query, code) gold-F1：0.7367；
- Target value over direct code：**−0.1008**。

Target确实依赖 code，但会遗漏已明确给出的答案；直接解析优于经过 Target。该路线会退化为 compressor自身完成 QA，冻结 Target不增加价值。

## 11. 统一、可由实验支持的结论

当前结果支持如下机制框架：

> Progressive compression for a frozen LLM is jointly constrained by faithful compression, Target compatibility, and observable stopping.

| 方法族 | Faithfulness | Target compatibility | Compactness / controllability | 实验性失败点 |
|---|---|---|---|---|
| V8 source text | 强 | 强 | 中等 | future rescue stopping不可识别 |
| Raw local intervention | 强 | 强 | 局部有机会 | repair稀疏且break不可预测 |
| Semantic compact evidence | 自动化不足 | 较强 | 强 | proposition validity / grounding |
| Internal/output stopper | 强 | 强 | 有oracle上界 | observability与重复调用成本 |
| Prefixable reordered text | 强 | 中等 | cache友好 | order改变导致质量下降 |
| Continuous latent | 不适用 | 弱 | 理论强 | frozen Target不会稳定消费 |
| Answer-like native code | task-statistic | 不稳定 | 极强 | direct parse优于Target，退化为QA solver |

V8 因而仍是当前合同下证据最完整的自动、faithful、Target-compatible progressive policy。项目尚未得到一个同时改善最终质量、累计 context 和真实 end-to-end compute 的 query-adaptive stopper。

## 12. 论文中可以和不可以写的表述

### 可以写

- V8 learned ordering 在预注册 development gate上达到 277/300 0.90、273/300 Complete，并暴露 surrogate-to-contract mismatch。
- Fresh 1,421-query trajectory、paired intervention、oracle ceiling和 query-grouped OOF结果可以作为机制实验。
- Adaptive stopping具有 9.42% Target-compute oracle headroom，但可部署 judge需要约96.6% FS rescue recall。
- Semantic、cache-compatible和latent alternatives分别受到 faithfulness、quality-order和Target compatibility限制。
- 上述失败不是单一 bug，而是一组由多轮独立审计支持的耦合约束。

### 不可以写

- “V8 已经通过最终测试”或“显著优于所有基线”。
- 把 oracle selector、oracle cutoff、gold code或 hindsight repair称为方法性能。
- 把 design-exposed internal300、606/611/581 cohorts称为 fresh confirmation。
- 把 latent positions直接称为节省的 text tokens。
- 根据 native-code existence声称 automatic compressor可学习。
- 根据负结果声称任何可能的 semantic/latent compressor在理论上都不可能成功。

## 13. 主要证据索引

## 13. Frozen-Target 外部压缩基线

### 13.1 LLMLingua-2 Screen-64

官方 `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` checkpoint在64个
确定性选取的design queries上，以冻结Qwen3-8B fresh配对评估。完整原始context
在requested keep=0.50（actual 0.5148）时，mean F1从0.9495降至0.5847，
0.90 success从53/64降至13/64。先经过V8 depth-10选择、再做global
LLMLingua-2，在相近压缩率下从0.9353降至0.6579，0.90 success从56/64
降至22/64。packetwise V8-wrapped arm为0.5811和22/64。更强压缩均进一步
下降，因此预注册Screen gate失败，决定为`STOP_LLMLINGUA2_AFTER_SCREEN64`。

fresh V8 control与历史缓存mean F1分别为0.93525与0.93509，排除了缓存漂移。
该结果可作为论文负基线；同时只记录、不触发重训的机制线索是：V8 coarse
filtering在token pruning前有益，global compression在约2x时比逐packet独立
compression更稳健。完整结果见
[report](results/v2_rank_then_cut/frontier_r1_hard_screen64/REPORT.md)与
[summary](results/v2_rank_then_cut/frontier_r1_hard_screen64/summary.json)。

## 14. 主要证据索引

- 总实验日志：[results/v2_rank_then_cut/RESULTS.md](results/v2_rank_then_cut/RESULTS.md)
- 实施与合同：[CODEX_FIDELITY_COMPRESSION_IMPLEMENTATION_PLAN.md](CODEX_FIDELITY_COMPRESSION_IMPLEMENTATION_PLAN.md)
- V8 run：[results/v2_rank_then_cut/v8_one_run_seed20260912](results/v2_rank_then_cut/v8_one_run_seed20260912)
- Fresh V8 chain：[results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain](results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain)
- Stopping feasibility：[results/v2_rank_then_cut/v17stop_c4_a0_required_stopper_feasibility/REPORT.md](results/v2_rank_then_cut/v17stop_c4_a0_required_stopper_feasibility/REPORT.md)
- V18 latent preflight：[results/v2_rank_then_cut/v18_l0a_latent_contract_preflight/REPORT.md](results/v2_rank_then_cut/v18_l0a_latent_contract_preflight/REPORT.md)
- V18 memorization：[results/v2_rank_then_cut/v18_l0b1_tiny_memorization/REPORT.md](results/v2_rank_then_cut/v18_l0b1_tiny_memorization/REPORT.md)
- Native-code existence：[results/v2_rank_then_cut/project_synthesis_a0_solution_existence/REPORT.md](results/v2_rank_then_cut/project_synthesis_a0_solution_existence/REPORT.md)
- Native-code consumption：[results/v2_rank_then_cut/native_code_a0_consumption/REPORT.md](results/v2_rank_then_cut/native_code_a0_consumption/REPORT.md)
