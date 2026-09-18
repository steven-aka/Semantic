# V10 科学与实现复审（训练前）

结论：**保留 mask-value + 精确规划方向，但当前冻结版本不通过训练前科学审查。** 不能依据 123 项单元测试通过，把它视为已经具备科学进入下一阶段的条件。主要阻塞是推理使用真实 active-level 信息；另有采样分布、训练目标与判定协议问题。V10 训练没有启动。

本次仅增加 CPU 审计及报告，没有更改模型、监督数据、训练参数、启动锁或冻结配置。`configs/v10_mask_value_probe.json` 中的 `READY_FOR_REVIEW_NOT_STARTED` 是上一次工程准备状态，不代表本次审查通过；本报告的审查结论优先于先前的就绪说明。

## 1. 哪些方向成立

- Target 对给定问题和源顺序渲染的 packet 子集产生固定输出，当前子集 fidelity 可以直接监督。预测这个量有意义，而且能利用已有 exact lattice，不需要新增 Target 推理。
- mask-value 不需要把历史作为输入来预测当前 fidelity；规划器仍维护最高已达层数，保留了非单调 fidelity 下所需的历史信息。这与 V8 动作标签需要历史并不矛盾。
- ordinal logit 只要求更高 fidelity 阈值的概率不超过更低阈值概率；没有强制“加入 packet 后 fidelity 必须提高”。这一约束正确。
- 冻结 V8 编码器、单 seed、一次开发集终点评测，是可控成本的诊断设计。但冻结表示能否保留预测 Target 细节所需的信息尚未证明。
- 本次把真实值代入新 DP，在已消费 development300 做正控：**300/300 完整成功，300/300 成本等于已有全局 oracle**。遍历这些 lattice 的加 packet 边，token 成本下降的边数为 0。这支持当前 DP 在本数据上的首次达标成本定义。

以上只验证规划器在正确值函数下有效，不验证学习出的值函数能足够准确。

## 2. 必须修正：真实 active_levels 进入解码

位置：`src/evaluation/mask_value_evaluation.py:75`、`:80`、`:97`。

当前实现从 `source['active_levels']` 获得层数，用于裁剪预测，并传入决定总顺序的 DP。这个字段来自 `near_optimal_chain_membership` 对真实 Target lattice 的分析，不是部署可见信息。train3163 中 672 题只有四层、2491 题有五层；development300 中分别是 66 与 234 题。

这不只是用于计算评价分母。审计构造的两 packet 反例是：预测 attained 为 `[0,4,5,0]`，token 成本 `[0,1,10,11]`，真实 attained `[0,4,0,0]`。告知规划器“真实只有四层”后顺序为 `[0,1]`、真实达到四层；固定五层时顺序为 `[1,0]`、真实一层也达不到。同样的模型分数，仅改变 oracle 提供的信息就改变最终成功率。

**要求：** 推理固定使用预注册五个 fidelity 阈值；`active_levels` 只能在训练标签/评测器内使用，不能进入预测或解码 API。对真实不可达 0.95 的 672 个训练样本，所有 mask 的 0.95 标签都已经知道为负，应纳入训练，而不是像当前 `ordinal_mask_value_loss` 一样全部屏蔽。新增测试应验证修改评价端 active-level 元数据不会改变同一输入的部署顺序。

## 3. 高风险：分层抽样与 0.5 阈值没有对齐

位置：`src/data/build_v10_mask_value_supervision.py`、`src/model/mask_value_head.py:78`。

| 阈值 | 完整训练 lattice 正例率 | 当前抽样后正例率 |
|---|---:|---:|
| 0.90 | 0.7159% | 20.3234% |
| 0.95（真实可达样本内） | 0.07614% | 3.62144% |

分层抽样保留稀有正例本身合理。但当前未加权 BCE 学习的是抽样分布下的判别关系，不能直接把 sigmoid 数值当成完整 lattice 上校准的达标概率。倍率分别约 28.4 和 47.6 倍，且抽样按“每题、attained class”进行，不能简单用一个全局正负比例改偏置就保证修复。

DP 会主动在 4096 个候选中寻找高预测值且便宜的集合，错误可能被搜索利用，而不是被搜索消除。例如若负例误报率为 0.1%，一题约 4000 个负 mask 的期望误报数仍约为 4；不需要独立性假设才能计算这个期望。这不是对当前 head 的实测误报率，只是说明全局 accuracy 很高也不足以保障路径。

**建议一次定清：** 保留分层采样，明确 loss 所估计的总体。若希望固定 0.5 具有原始分布含义，应采用已知入样概率校正，按 query 平均，再按固定五个 anchor 平均。对某类 N 个状态、抽 m 个、强制保留 r 个端点，端点入样概率为 1，其余为 `(m-r)/(N-r)`；权重为其倒数。不得把端点也简单赋为 N/m。当前 class_population 足以推导这些概率。校正解决目标分布不一致，不保证模型校准或最终达标；尤其要监测稀有正例召回。

同时报告完整 lattice 的逐层 precision/recall，以及**最终选中路径/预测 cutoff 上**的误报率和成本，避免只看总体 classification accuracy。预测截止点的诊断结果不能代替现有排序 gate，但可以识别 DP 是否在利用值函数误差。

当前 loss 还将全 batch 的有效 mask-anchor 项直接平均：每题贡献 160--480 项，原始项数相差 3 倍，并非 example-balanced。不同 query 在当前采样器中的梯度权重会受可达层数及稀有类状态数影响。应显式定义 query 权重，不能只用“训练状态很多”解释覆盖充分。

## 4. 门槛通过意味着什么

位置：`src/evaluation/mask_value_evaluation.py:101`--`:102`。

当前评测使用 `best_prefix_nested_chain(exact, order, levels)`，仍由真实 Target lattice 为预测顺序挑选全局最佳截断位置。**这是已有排序阶段合法的诊断口径，但不是端到端压缩结果。** 移除前述解码泄漏后，可以继续保留这个口径与 V8 比较；不能声称 value head 已经解决实际停止位置或 fidelity 保证。

科学的后续顺序应明确写入 V10 协议：

1. 修复推理与目标定义，保持既有开发 gate：0.90 >=282/300、完整轨迹 >=0.90、可行 regret <=0.03。
2. 在训练角色内检查数值稳定性和学习情况，固定训练预算与唯一终点，然后只做一次 consumed-development gate。400 步约两个 epoch 只是预算，不能保证收敛。
3. 通过开发 gate 后，冻结**同一个 checkpoint 与同一个 decoder**，预注册一份真正未使用的 confirmation population、资格规则、样本数和全部判定；不重训、不根据确认结果选择新阈值。
4. 一次新确认沿用项目已有统计要求：每个 anchor 的单侧 Clopper–Pearson 下界，经五层 Bonferroni（各 alpha=0.01）后 >=0.90，并检查 active-contract >=0.97、完整轨迹 >=0.90、regret <=0.03。
5. 通过新确认才进入 cutoff/校准阶段。calibration300 与 final-test300 继续保留给其预注册用途。

用仓库已有函数计算，282/300 的 alpha=0.01 CP 下界为 0.9002015，277/300 为 0.8800377。但**反复使用的 development300 上的这个计算不能恢复新鲜验证的统计含义**。0.95 的实际可达样本数可能小于 300，其通过计数必须按实际分母另算，不能照抄 282。

另需收紧负结论：V9-A/B 的结果只停止各自测试过的冻结表示、预算与损失组合；它们不能证明所有 cost learning 或覆盖方法无效。同理，V10 一次 400 步失败只能停止这份配置，若训练角色也没有学好，不能据此否定整个 mask-value 假设。

## 5. 代码判定与工程准备缺口

`src/evaluation/v10_mask_value_decision.py:6` 没有核验 `complete`、总题数和逐层分母。审计传入 `complete=false`、282/400，当前函数仍返回 GO；regret 为 None 时抛 TypeError。当前固定开发数据恰好是 300 题，但独立评测入口也调用这个函数，因此它并不是严格的 282/300 判定。应让不完整/分母错误的结果拒绝判定，对无可行轨迹给出明确失败，不允许静默 GO。

配置哈希检查覆盖了新代码和部分 checkpoint，但未覆盖实际读取的 LoRA adapter 权重、部分共享 encoder/loader/planner，以及启动器内硬编码的训练参数。需要把参数来源统一到配置，记录完整依赖与 adapter 哈希，并在训练开始时保存协议快照。

单一“评测终点”不要求只能在训练末尾保存文件。增加只用于恢复的模型/optimizer/scheduler/RNG 状态不会引入开发集 checkpoint 选择，可降低中断后重复训练的成本。

显存测试没有运行。冻结底座有助于节约显存，但不足以证明 batch16 一定可用；纯前向/不执行 optimizer.step 的资源检查属于准备工作，不等于实验训练，之前把二者等同并不准确。本次依然没有执行实际 4B GPU 检查。

## 6. 节省计算的优先改进

底座与 packet encoder 完全冻结并使用 eval 模式，适合把每题的 12 个 packet 向量和 1 个 question 向量提前缓存。BF16、512 维下，train3163 的裸向量仅约 42.1 MB，development300 约 4.0 MB（不含元数据）。缓存一次后即可释放 4B 模型，只训练 5.26M 参数 head，避免约两轮重复编码，并把显存和推理预算留给 mask head。

缓存必须按数据、tokenizer、底座、LoRA、encoder 及长度设置做哈希；训练与开发缓存分开，训练器不接收开发标签。这是计算优化，不是提升预测质量的保证。编码分批与缓存数值一致性也需验证。

## 7. 审计产物与复现

- `src/evaluation/v10_pretraining_audit.py`：CPU-only，不加载模型、不创建 optimizer；统计训练分布、复现解码信息泄漏和 gate 反例，并执行 300 题真实值规划正控。
- `results/v2_rank_then_cut/v10_pretraining_audit.json`：完整结果。
- 运行：`.venv/bin/python -m src.evaluation.v10_pretraining_audit`。
- 额外只加载本地 Qwen3-8B tokenizer，对 development 前三题的 12,288 个 mask 从 packet 文本重算 token，检查缓存成本是否可在不使用 Target 标签时获得；结果记录于 JSON 的 `label_free_token_reconstruction_check`（主审计命令不执行此附加检查）。

本次审查没有给出“V10 一定过线”的承诺。现有证据支持继续完成上述最小修正，再做一次有边界、可解释的实验；不支持直接运行当前冻结版本并把通过开发 gate 当作问题解决。
