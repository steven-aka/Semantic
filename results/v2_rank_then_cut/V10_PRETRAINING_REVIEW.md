# V10 训练前审查

> 后续科学复审发现阻塞问题；本文件的工程准备记录不代表审查通过。请优先阅读 `V10_SCIENTIFIC_REVIEW.md`。训练仍未启动，冻结配置尚未按复审建议修改。

状态：`READY_FOR_REVIEW_NOT_STARTED`。截至 2026-09-17，没有 V10 训练进程、优化器步进、checkpoint 或开发集结果。

## 为什么做这一次实验

V8 的 step250 在开发集达到 fidelity-0.90 277/300、完整轨迹 273/300、regret 0.02793。V9-A 的代价损失和 V9-B 的覆盖补充分别只有 274/300 和 275/300；三者失败集合中有 20 个共同样本，事后逐样本 oracle 选择也只有 280/300。因此，继续改动作损失、beam 宽度或固定历史池，没有证据能达到 282/300 门槛。

V8/V9 学的是“下一步选哪个 packet”，但真正的 Target fidelity 是当前 packet 子集的非单调函数。V10 直接学习当前 12-bit mask 是否达到 0.60/0.70/0.80/0.90/0.95，再在预测出的完整 4096-mask lattice 上做精确动态规划。这是在不再次训练 4B 底座的前提下，对现有 surrogate mismatch 的最直接检验。

## 已准备内容

- 训练数据：3163 个原训练角色样本、253,973 个分层抽样 mask。每个样本的每个实际 attained class 最多保留 16 个 mask，稀有类全部保留；mask 0 与 mask 4095 在各自类别配额内强制保留，保证训练覆盖推理起点和终点的特殊池化分支。
- 开发数据：300 个已消费开发样本的完整 1,228,800 个 mask，只在训练完成后评测一次。
- 模型：冻结 V8 step250 的 Qwen3-4B、LoRA、packet/question encoder 和原顺序 head，只训练一个两层、512 维、8 头、5,258,757 参数的 mask-conditioned ordinal head。
- 输出约束：五个 ordinal logit 由结构保证单调，阈值固定为 0.5，不做开发集校准。
- 解码：枚举预测 4096 个 mask，以精确 subset DP 最大化达到的层数，再最小化首次跨越各层的累计 token；输出仍是一个 12-packet 总顺序及其嵌套前缀。
- 优化：单 seed 20260912，batch 16，400 步，约 2.02 epoch；AdamW、lr 2e-4、40 步 warmup、cosine、clip 1.0。head 参数和 AdamW 状态保持 FP32，前向使用 BF16 autocast。
- 选择：没有中间 checkpoint，没有开发集 early stopping，只保存并评测 step400。
- 验证：完整单元测试 123/123 通过；训练/开发 ID 不重叠、逐样本分层计数、开发完整 lattice、exact 文件存在性及数据 SHA-256 均通过。

## 数据边界与判定

训练只使用原 train3163 的 Target exact lattice。开发角色仍是已经消费的 development300；它只承担这次诊断 gate，不作为新鲜证据。calibration300、final-test300 和任何 locked role 都没有读取。

预注册判定为：fidelity-0.90 至少 282/300、完整轨迹至少 0.90、可行轨迹平均归一化 regret 不超过 0.03，三项同时满足才进入下一步。如果通过，冻结本次同一个 head，在一个新冻结的 target-blind confirmation role 上直接评测，不重新训练。fidelity-0.90 为 279--281 时判为不确定，不打开新角色；不超过 278 或任一次要门槛失败则停止该假设。

## 主要风险

1. mask 分类准确率与最终轨迹门槛仍可能不完全一致；完整 lattice 的 DP 能减少局部错误，但不能消除系统性高阈值误判。
2. fidelity-0.95 正例在总体 lattice 中极稀有。分层抽样已保留训练角色中 7,441/7,769 个 class-5 状态，但它仍可能是最难泛化的一层。
3. batch 16 利用冻结底座不保存反向图带来的显存余量，但尚未做真实 4B GPU 内存 smoke。按用户要求，GPU 动作和训练均停在审查前；批准后应先运行启动脚本的哈希检查，再在空闲 A6000 上启动。
4. 这是 consumed-development probe。即使通过，也只能授权一次新确认，不能直接宣称方法已经在未见数据上成立。

冻结配置在 `configs/v10_mask_value_probe.json`。启动脚本 `scripts/64_launch_v10_mask_value_probe.sh` 有双重锁：必须显式传入 `--execute`，且配置状态必须人工改为 `APPROVED_TO_RUN`；当前状态会拒绝训练。监控入口是 `bash scripts/65_monitor_v10_mask_value_probe.sh`。
