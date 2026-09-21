# Frontier Compression Reproduction Plan

## 1. 研究目标与本阶段定位

最终目标保持不变：

> 给定 query 和原始 context，学习一个面向冻结 Qwen3-8B Target 的渐进式上下文压缩/揭示策略，在质量约束下改善真实的质量–成本 Pareto frontier。

本阶段不是做一张跨论文 leaderboard，也不要求外部方法与 V8 使用完全相同的表示或训练方式。它是一组机制实验，用来回答：在本项目的 QAMPARI 多答案、分散证据环境中，信息应当以什么形式压缩，才能同时具备紧凑性、Target 可消费性和渐进可扩展性。

当前 V8 是本项目内证据最完整的自动文本基线，但尚未通过最终独立确认。所有新方法均先使用 train/design-exposed 数据；lineage-clean 581、internal、development 和 fresh confirmation 继续封闭。

## 2. 对原指导的审查结论

### 2.0 Frozen-Target 资格定义

本项目只正式复现满足以下条件的方法：

1. Qwen3-8B Target 的全部原始参数冻结；
2. 不在 Target 上训练 LoRA、adapter、prefix tuning 或新增可训练层；
3. compressor、retriever、encoder、bridge 可以训练，但必须是 Target 外部模块；
4. Target 可以通过标准 `input_ids`、`inputs_embeds` 或合法 layerwise KV prefix 接收压缩表示；
5. 任何必须修改 Target decoder attention 规则或训练 decoder adapter 的论文方法，不进入正式候选。

“基础权重冻结但训练 Target LoRA”不算 frozen Target。单独的 compressor 可以复用同架构的 Qwen 权重并训练 LoRA，只要部署用的 Qwen3-8B decoder 是另一个参数完全冻结的实例。

### 2.1 应保留的部分

1. 保留统一 frozen Qwen3-8B、prompt、解析器和 QAMPARI scorer。
2. 将 paper-mechanism sanity 与 Qwen3 target-port 分开。移植失败不能直接解释为论文机制失败。
3. 把 V8 ordering 作为可复用资产，分别测试 native compressor 与 V8-wrapped compressor。
4. 同时报告表示长度、端到端成本、多档质量和轨迹非单调性。
5. 对 QAMPARI 做 distributed-evidence 分层，并保留 query-only、shuffled-context 反事实。
6. 固定预算 representation frontier 后，才研究 per-query budget selection 或 stopping。

### 2.2 必须修正的部分

1. **不同时完整复现九种方法。** 这会把主要成本花在依赖、训练配方和接口移植上，且方法间差异太多，无法识别因果机制。
2. **统一的是评估记录，不是 Target 输入语义。** `input_ids`、`inputs_embeds` 和 layerwise `past_key_values` 需要不同位置编码、attention mask、cache 格式和服务路径，不能由一个无类型的 `to_target_inputs()` 掩盖差异。
3. **PISCO 从正式名单移除。** 原论文为 decoder 训练独立 LoRA，并明确报告 frozen decoder 不足以获得满意结果，因此不符合本项目的 frozen-Target 合同。它的 sequence-level KD 只能作为训练目标启发，不能作为待复现方法。
4. **500xCompressor 先做可复现性门禁。** 它的 decoder 是原始冻结 LLM，符合合同；但公开仓库目前注明模型/数据未公开，在 Qwen3 上需要自行训练时，它属于 mechanism port，不是 checkpoint reproduction。
5. **SAC 有条件符合。** 论文训练的是独立 encoder LoRA，Target decoder 参数保持冻结；双向 attention 和 anchor embedding只存在于 encoder端。移植时必须保持这种 encoder/decoder隔离，不得修改Qwen3-8B Target前向。
6. **RAM 暂不列入正式复现。** 当前公开说明包含联合 answer-generation训练，但不足以确认Qwen3 Target decoder可完全冻结并原样移植。在确认论文训练参数边界前，只保留其 raw-core/compressed-tail思想。
7. **xRAG 加入正式 soft 候选。** 它只训练 modality bridge，retriever和语言模型均冻结，严格符合合同；其公开代码和checkpoint也比500x更适合先做机制sanity。
8. **KV cache eviction 与 semantic prefix compression 分开。** 删除已计算 KV 的 serving 方法主要优化内存/吞吐；生成能替代原始上下文的 KV prefix才回答本项目的语义压缩问题。
9. **High-norm 只能做诊断。** 在一个尚不能完成 tiny memorization 的表示上扫缩放系数，无法证明高 norm 是缺失机制。只有已训练可消费表示才允许做 norm intervention。
10. **FlexComp 只在一个 fixed-budget backbone 已经成功后启动。** 否则 multi-budget training只会把一个失败 carrier扩展到多个预算；并且只能套在符合上述 frozen-Target资格的backbone上。
11. **RECOMP 的预训练 checkpoint 与 QAMPARI 域不匹配。** 它可作为 text-semantic control；若重新训练，则必须称 target-domain port，并把抽取式与生成式分开报告。

## 3. 五个待回答的机制问题

| 编号 | 科学问题 | 最小充分对照 |
|---|---|---|
| Q1 | V8 packet 内部还有多少可用的文本压缩空间？ | V8 vs LongLLMLingua/LLMLingua-2 native 与 V8-wrapped |
| Q2 | V18 的失败主要来自表示生成方式还是 carrier？ | xRAG式 bridge positive control；随后同一encoder下 embedding-prefix vs layerwise-KV-prefix |
| Q3 | source-native anchor 是否比 arbitrary latent 更易被 frozen Target 消费？ | 同预算、同训练目标下 anchor/KV vs arbitrary latent/KV |
| Q4 | QAMPARI 更适合纯压缩还是 raw-core + compressed-tail？ | 最佳纯 carrier vs V8-guided hybrid |
| Q5 | multi-budget training 能否产生稳定、近似嵌套的 progressive frontier？ | fixed-budget specialists vs one-model multi-budget training |

每一阶段只为回答其中一个问题。若相应 ceiling 或实现门禁失败，后续依赖该机制的实验不启动。

## 4. 统一软件接口

不要用一个返回任意对象的接口。使用带类型的表示：

```python
@dataclass
class TextRepresentation:
    input_ids: Tensor
    attention_mask: Tensor
    source_map: list

@dataclass
class EmbeddingRepresentation:
    inputs_embeds: Tensor
    attention_mask: Tensor
    position_ids: Tensor

@dataclass
class KVRepresentation:
    past_key_values: object
    prefix_attention_mask: Tensor
    prefix_position_count: int
    cache_bytes: int

class ContextCompressor(Protocol):
    def prepare(self, query, context, packets, v8_order): ...
    def compress(self, budget) -> TextRepresentation | EmbeddingRepresentation | KVRepresentation: ...
    def stats(self) -> dict: ...
```

每个 backend 必须各自通过以下 no-op/identity tests：

- 原始 text path 与 canonical runner 一致；
- token embedding lookup 经 `inputs_embeds` 输入时，与相同 `input_ids` 的 logits/生成结果在冻结容差内一致；
- 标准 prefill 得到的 KV 经 cache path 继续生成时，与一次性前向一致；
- position ids、RoPE offset、GQA head layout、padding side、chat template 与 canonical manifest 一致；
- batch=1 和正式 batch 配置分别记录，不默认它们等价。

若 KV identity test 不通过，任何 semantic-KV 训练均停止。

## 5. 统一数据与防泄漏合同

### 5.1 三层数据

1. **Screen-64**：固定的 design-exposed 64 queries，用于安装、接口、明显失败检测。
2. **Design-256**：与 Screen-64 不重叠的 design-exposed 256 queries，用于形成初步 frontier。
3. **Train-heldout**：从 canonical train 按 query 分组冻结 train/validation；learned compressor 的所有组件和上游 scorer 都不得见 validation query。

581、internal、development、confirmation 在选出单一候选并冻结协议前不得读取。

### 5.2 推理输入

部署时 compressor 只能读取 `(q, C)` 及从二者确定性导出的 packets/V8 scores。Gold answers、Target answers、oracle actions 和 fidelity labels只能作为训练监督或离线分析字段。

### 5.3 Teacher 合同

Soft/KV 方法第一轮统一使用 frozen Qwen3-8B 在 **V8 depth-10 context** 上的固定 greedy output作为 sequence teacher。这样测试的是能否压缩当前可靠 operating point，而不是追逐 full-context 的非单调输出。另报 full-context teacher 作为诊断，不同时调参。

## 6. 统一评估合同

每条 `query × method × budget` 至少保存：

- qid、method、variant、seed、budget 和 representation type；
- Target 原生 positions、text tokens、memory slots、KV bytes；
- compressor、Target prefill、Target decode、总 wall time和 peak VRAM；
- Target 调用次数；
- raw answer、parsed answer、gold F1、precision、recall；
- `pass_060` 至 `pass_095`、Complete；
- source attribution/answer-mention diagnostics（只作分析）；
- soft/KV 的 norm、effective rank 和 layerwise cache size。

成本不压成一个无来源的标量。至少给出三张 Pareto 图：

1. `Target positions / KV bytes` 对质量；
2. batch=1 与正式吞吐场景下的真实延迟对质量；
3. compressor FLOPs/latency + Target latency 对质量。

主质量指标是 mean QAMPARI F1、五档成功率与 Complete。Gold-answer mention recall只是诊断，因为别名、释义和生成式压缩会使字符串 mention 与可回答性不完全一致。

所有主比较使用 paired query difference 和 bootstrap 95% confidence interval。64-query screen 不做显著性结论。

## 7. Progressive 合法性

一次性 compressor 获得好的固定预算结果，并不自动满足主线要求。只有满足下列条件之一，才能称为 progressive representation：

1. 最大表示只构造一次，小预算表示是其严格前缀或确定性子集；
2. 每个 packet 只压缩一次，V8决定逐档加入哪些packet，渲染继续使用canonical source order，已有表示不被改写；
3. cache/anchor 状态具有明确、可验证的 nested extension 操作。

对每个预算序列计算 SS/SF/FS/FF 和 `P(late fail | early success)`。比较 SF 时必须匹配 early-success prevalence；否则更弱的早期模型可能因早期成功更少而获得虚假的低 SF。

## 8. 执行阶段

### R0：Artifact 与 Target 兼容性审计（零 Target 训练）

为每个候选记录：官方代码/权重/许可证、原 backbone、是否需要修改 Target attention、是否支持 Qwen3/GQA/RoPE、训练数据与预计 GPU 成本。

分组：

- 可直接 screen：LongLLMLingua、LLMLingua-2；
- 可作 text control 但需域适配判断：RECOMP；
- 有公开 frozen-Target positive control：xRAG；
- 需 mechanism port：500xCompressor；
- frozen Target、独立 modified encoder：SAC；
- 仅在前置成功后：FlexComp；
- decoder冻结边界未确认或不符合合同：PISCO、RAM不进入正式复现；
- 无可审计官方资产或与前述机制重复：暂不启动 SARA/high-norm独立复现。

R0 输出一张 compatibility matrix。没有通过 identity/runtime smoke 的方法不得进入 Screen-64。

### R1：Hard/Text Frontier

固定方法：

1. canonical raw context；
2. V8 原始文本与固定 schedules；
3. LongLLMLingua native；
4. LongLLMLingua V8-wrapped；
5. LLMLingua-2 native；
6. LLMLingua-2 V8-wrapped。

预算为实际 Qwen3 token budget，不以 compressor 自报比例为准。Screen-64 取近似 `2x/4x/8x` 三点；通过后 Design-256 增加 `16x`。`6x` 和 `32x` 只有当相邻点显示可用区间时再加。

V8-wrapped规则：每个packet独立、只压缩一次，question/instruction不压缩；V8只决定各depth选择的packet集合，最终文本仍按canonical source order渲染。这与项目现有V8语义一致，也避免C3已经观察到的文本顺序扰动。它是一个受控progressive arm；native全上下文压缩只评固定预算，不冒充nested trajectory。

RECOMP 不列为 R1 必跑项。只有上述 extractive 方法没有形成可用 frontier，或需要区分“token deletion”与“answer-oriented textual summary”时，才增加一个 RECOMP extractive arm和一个 abstractive诊断 arm。

**R1 继续门槛（Design-256 前冻结）：** 在至少一个 `>=4x` 实际压缩点上，paired mean F1下降不超过2个百分点，且 0.90/Complete 各下降不超过3个百分点；或在质量差不超过上述界限时，真实端到端延迟改善至少20%。仅 SF 改善不能单独通过，除非 matched early-success下质量不低于 V8 相邻 operating point。

### R2：Frozen-Target Soft/KV 可行性与 carrier 因果对照

此阶段不是并排复现四篇论文，而是一个受控实验：

1. 先用xRAG官方Mistral配置完成paper-mechanism sanity，确认bridge + frozen decoder链路可运行；
2. 再做Qwen3 xRAG-style bridge tiny memorization；Qwen3-8B参数全冻结，只训练外部retrieval encoder到Target表示空间的bridge；
3. 500x-style port只有在官方训练配方和代码能被审计后启动；
4. 在完全相同的 encoder、teacher、训练 queries、memory budget和优化步数下，仅改变输出 carrier：
   - A：Qwen3 `inputs_embeds` prefix；
   - B：Qwen3 layerwise KV prefix。
5. 预算只取 `4x/8x/16x`；先32-query memorization，再query-heldout。

Tiny gate 同时要求 teacher-forced loss下降、free-generation F1大幅高于 V18 的0.0313、query-only/shuffled-context明显退化。仅 NLL下降不能通过。

若 KV 胜过 embedding，结论限定为“在该受控 port 中 carrier 是关键因素”；不能将不同论文、不同 encoder 的分数差直接归因于 carrier。

只有此阶段出现可消费 soft/KV 表示，才做 norm scaling诊断。缩放不作为模型选择超参。

### R3：Anchor 与 Hybrid

R3a 先测试 source-native anchor是否提高Target compatibility。SAC式双向attention和anchor embedding只能存在于独立encoder中；canonical Qwen3-8B decoder仍使用原始causal forward和冻结参数。若移植需要修改Target decoder，则该实现不合格并停止。

R3b 使用R2/R3a最佳compressed carrier构建两组我们自己的hybrid对照，不把它称为RAM复现：

- native hybrid relevance scorer；
- V8-guided hybrid：V8 top-`m` packets保留 raw text，其余压成 memory，`m ∈ {2,4,6}`。

Hybrid 的关键对照是同一总成本下的 pure text、pure compressed 和 hybrid，不是只与 full V8 比。若 raw core 与 compressed tail 各自由不同模型训练，需将额外参数、压缩器计算和 peak memory完整计入。

### R4：Multi-budget / Progressive Training

只有 R1–R3 中至少一个 fixed-budget 方法通过质量和成本门槛，才在该单一 backbone 上做 FlexComp式 Matryoshka training。预算从已观测 Pareto 点选择，不预先机械使用 `{4,8,16,24,32}`。

先验证一个模型是否能匹配相同预算的 fixed specialists，并验证 budget states 的嵌套/可追加性。之后只做 single-pass K predictor：

`g(q, C, tau) -> K`。

不使用需要多次 Target generation 的 confidence cascade，除非它在真实 Target 调用成本下仍有正收益。

### R5：Train-heldout 与后续封闭集

每个机制家族最多留下一个候选。统一在 query-heldout 上报告完整 Pareto、分散证据分层和反事实。只有单一预注册候选在 train-heldout 形成新的质量–总成本 Pareto 点，才冻结方法并申请读取下一层 research gate。

## 9. Distributed-evidence 与反事实分析

按原始 context 中含 gold aliases 的 packet 数预注册 low/medium/high dispersion strata，例如 `1–2 / 3–5 / 6+`。同时报告：

- gold alias覆盖；
- answer-bearing packet覆盖；
- 最终 answer recall；
- 各层 mean F1、0.90和Complete。

这些 strata 只能由评估脚本使用，不能进入 compressor。

所有 learned semantic方法至少跑：

- query-only；
- query + shuffled other-query context；
- correct query + correct context。

若 shuffled 与 correct 差异很小，该模型不具备 context consumption证据，即使绝对 F1尚可也不能作为主线候选。

## 10. 方法选择规则

最终不选“平均 F1最高”的单点，而选满足以下条件的候选：

1. 在至少一段预算区间形成非支配点；
2. 改善在 query-heldout 和多数 bootstrap resamples中存在；
3. high-dispersion strata没有灾难性崩溃；
4. context/shuffle controls证明模型使用了原始 context；
5. 若声称 progressive，表示确实 nested，且 SF/FS 分析使用正确标签和 matched prevalence；
6. 总成本包含 compressor、Target、KV memory和重复调用。

## 11. 最短执行路径

为了最快得到有决策价值的结果，实际顺序冻结为：

1. R0 compatibility matrix + 三种 Target backend identity tests；
2. R1 LongLLMLingua/LLMLingua-2 的 native 与 V8-wrapped Screen-64；
3. 仅对 R1 winner 做 Design-256；
4. R2 xRAG frozen-decoder positive control，再做单一architecture的embedding-vs-KV carrier experiment；
5. 只有 R2 有正结果才做 anchor/hybrid；若 R1 已经强势形成 frontier，可直接先做 raw-text progressive方案，不等待 soft分支；
6. 只有一个 fixed-budget compressor成功后才做 multi-budget progressive training。

该顺序最大程度复用 V8 packets、ordering、canonical Target runner、cached path、scorer和成本记录，同时避免重新陷入“每篇论文各自训练一套、结果却无法归因”的问题。

## 12. 允许得出的结论

这一阶段可以回答五个机制问题，并可确定下一代方法的 carrier。它不能直接证明最终方法优于全部文献，也不能把 paper sanity的结果当作 Qwen3/QAMPARI结论。

若 hard/text方法胜出，下一代是 V8 ordering + within-packet pruning + nested reveal。若 KV/anchor胜出，下一代是 V8-guided progressive semantic memory。若 hybrid胜出，下一代是 raw evidence core + compressed tail。若所有移植都失败，则说明当前 frozen Qwen3-8B 对非原生表示的兼容性或 QAMPARI分散证据需求构成结构性限制，V8 fixed schedule继续作为主基线，研究应转向训练 Target-aware compressor所需的监督与接口合同，而不是继续扩 selector或 stopper。
