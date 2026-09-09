# Fidelity-Constrained Successive Semantic Compression

本仓库按 `CODEX_FIDELITY_COMPRESSION_IMPLEMENTATION_PLAN.md` 构建。当前实现范围是 **V0 + V0.1--V0.5 有界诊断，以及 M0 根因验证**；遵守计划中的 STOP POINT，尚未实现或启动 QLoRA/V1 训练。

## 当前实验进度

HotpotQA validation 已下载到 `data/raw/hotpot_validation.parquet`。目前已准备并审计 100 条候选数据、10×5 smoke 集与 30×6 controlled pilot 集。真实 full-context benchmark、15 条 eligible controlled packet、每题 `3^6=729` 状态的 exact search 均已完成。

原 V0 的 supporting-unit 代理指标使所有 `state>=1` 的支撑单元都得到满分，导致前沿不可辨识。V0.1 将 15 题中的 35 条原始 gold supporting sentences 精确映射回当前单元，并用同一冻结 Qwen3-8B 分别审计已有 gist 与 gist+residual 是否完整支持该事实。主网格 0.60–0.95 仍保持不变；另输出覆盖实际可达事实召回断点的敏感性网格。敏感性网格上有 12 次码率/状态切换，但全部为嵌套切换，故当前零结构差只能作描述性结果，不能证明嵌套约束没有代价。详情位于 `results/v0_1/diagnostics*.json`。

V0.2 保持问题、答案、模型、三态表示和 fidelity 定义不变，只把 gold supporting sentences 拆成可独立寻址的语义单元，构成 17 题的 oracle 结构压力测试。12,393 个状态全部完成后，预注册主网格首次出现唯一的非嵌套独立最优切换：结构税为 2 token（5.88%）；自动可达断点网格在 5 个真实样例上产生 23 个正差点，最大 47 token（22.38%）。这说明结构差已经可辨识，但 gold-aware 分段不能作为最终无偏基准。完整结论见 `results/v0_2/RESULTS.md`。

V0.3 是冻结后只运行一次的 sentence-atomic + label-free BM25 验证。30 个全新样本、180 个 packet 和 21,870 个 exact states 全部完成且工程验证通过，但科学 gate 为 **NO-GO**：top-6 gold-fact coverage 为 70.83%（要求 80%）、主网格严格码率切换率为 13.89%（要求 20%）、0.60→0.90 平均事实召回增益为 0.111（要求 0.15）。同时仍观察到一个 41-token（21.69%）正结构差点，说明无 gold 边界本身可以激活结构税；主要瓶颈是单跳 BM25 无法覆盖多跳事实。详见 `results/v0_3/RESULTS.md` 与 `results/v0_3/gate_result.json`。V1 继续停止。

随后完成了预注册的纯 CPU Retrieval R0：在排除前序 130 个 ID 后，用 500 题开发集比较三种固定的无标签词法/标题图策略，并只在 200 题锁定测试上评估开发集胜者。linked-title-pair 将开发覆盖率从 global BM25 的 66.37% 提升至 75.86%，但锁定测试只有 376/497=75.65%，未达到 80%。因此按规则输出 `STOP_RETHINK_N6`，没有构造 V0.4、没有启动 GPU。详情见 `results/retrieval_r0/RESULTS.md`。

Retrieval R1 随后在排除全部 830 个已用 ID 的新锁定集上，用冻结 Qwen3-14B 只读取 question/context 选择六句。它取得 430/481=89.40% gold-fact coverage，3/200=1.5% 格式回退，同时通过 80% 覆盖和 2% 回退门槛。由此生成的 V0.4 30 题子集覆盖 65/71=91.55%。完整 180 packets 与 21,870 states 均已运行并独立验证，但 V0.4 科学 gate 仍为 **NO-GO**：主网格严格码率切换仅 2/41=4.88%，nested reuse 为 50%，0.60→0.90 平均事实增益为 0.0583。主要矛盾已经从检索转为离散 fidelity 在 0.80/0.90/0.95 上的平台化。详见 `results/retrieval_r1/RESULTS.md` 与 `results/v0_4/RESULTS.md`。

最后一次有界修复 V0.5 按计划的监督条件，只从 R1 剩余冻结池中取 full-context F1≥0.8 的前 30 题。30/30 均可答、检索覆盖 67/72=93.06%、nested reuse 100%，但主网格严格码率切换仍只有 5/82=6.10%，0.60→0.90 平均事实增益只有 0.0833。它排除了检索不足与不可答样本两个解释，确认当前三态 packet 与 hard-min fidelity 在固定五档上信号过稀。按预注册规则停止继续修改 V0；详见 `results/v0_5/RESULTS.md`。

M0 随后从根因上处理 fidelity 信号瓶颈。连续 YES/NO margin 与答案 NLL
经真实 smoke 后被否决：置信度不能可靠代表语义保留。实验转到官方
QAMPARI 开发集的十答案原子任务，并采用确定性 lossless additive
partition，使 gist+residual 精确重建源证据。30 题、180 packets、21,870
states 已完整运行：严格 rate 增长 104/104，非嵌套切换 16 次，nested
reuse 87.5%，0.60→0.90 atom recall 增益 0.40，结构差不再恒为零。

最终联合 gate 仍为 **NO-GO**，唯一失败项是 target–benchmark
answerability：仅
25/30=83.33% 样本在任一状态达到 alias-aware list F1≥0.9，低于冻结的
90% 门槛。失败题审计同时发现模型八项枚举遗漏与受控证据/标签不闭合，
因此不能把失败单独归因于压缩算法或模型能力。门槛未在观察结果后修改，
也未运行 QAMPARI test target inference
或 V1/QLoRA。权威结果见 `results/m0_qampari/RESULTS.md` 与
`results/m0_qampari/lossless_dev_gate_result.json`；完整根因、已解决问题与
未解决问题见 `FIDELITY_SIGNAL_BOTTLENECK_PLAN.md`。

## 当前可复现实验流程

项目内 `.venv` 使用 Python 3.11；脚本默认调用 `.venv/bin/python`，也可通过 `PROJECT_PYTHON` 覆盖。正式 rate 统计始终使用 `Qwen/Qwen3-8B` tokenizer；`WhitespaceTokenizer` 只用于无依赖单元测试和 dry run，其输出不得作为论文结果。

GPU 脚本默认通过 `scripts/cuda_env.sh` 选择物理 GPU 3（RTX A6000 48 GB）。共享服务器上应在每次运行前检查空闲显存，并可用 `CUDA_VISIBLE_DEVICES` 覆盖。Target 的 full-context 和 exact-search 统一使用 0.65 显存占比；两个脚本默认离线读取 `models/Qwen3-8B`，避免正式实验中发生模型漂移或意外联网。

1. 准备 HotpotQA distractor 数据并切成无重叠语义单元：

   ```bash
   CANDIDATE_LIMIT=100 N_EXAMPLES=10 N_UNITS=5 bash scripts/01_prepare_data.sh
   ```

2. 先运行冻结的 Qwen3-8B 全上下文基准：

   ```bash
   bash scripts/02_run_full_context.sh
   ```

   `results/full_context.jsonl` 中 `eligible_for_supervision=true` 表示 full-context F1 ≥ 0.8；后续 V0 搜索读取自动导出的 `data/units/hotpot_exact_pilot_eligible.jsonl`。最终报告仍须保留完整测试集。

   10-query smoke 通过后，运行 30-query controlled baseline：

   ```bash
   bash scripts/02_run_full_context_controlled.sh
   ```

   结果和监督候选分别写入 `results/full_context_controlled.jsonl` 与 `data/units/hotpot_controlled_pilot_eligible.jsonl`。

3. 用冻结的 Qwen3-14B 逐单元生成 query-independent `gist + residual`，自动验证并缓存：

   ```bash
   bash scripts/03_generate_packets.sh
   ```

   默认先处理 5-unit smoke eligible 集并写入 `data/packets_smoke`。受控 6-unit 数据使用独立缓存，避免相同 `example_id` 的不同分段互相覆盖：

   ```bash
   EXAMPLES=data/units/hotpot_controlled_pilot_eligible.jsonl \
   PACKET_DIR=data/packets bash scripts/03_generate_packets.sh
   ```

4. 对 5 单元 smoke test 枚举 `3^5=243` 个状态；正式 6 单元 pilot 使用 `N_UNITS=6`，每题 `3^6=729` 个状态。所有状态按批次送入 vLLM：

   ```bash
   bash scripts/04_exact_pilot.sh
   ```

   smoke 通过后运行 controlled exact search：

   ```bash
   EXAMPLES=data/units/hotpot_controlled_pilot_eligible.jsonl \
   PACKET_DIR=data/packets OUTPUT_DIR=results/exact_search \
   bash scripts/04_exact_pilot.sh
   ```

5. 提取独立最优前沿，用动态规划求最小累计 rate 的嵌套链，并输出结构差距和图：

   ```bash
   bash scripts/05_analyze_v0.sh
   ```

   smoke 搜索对应 `bash scripts/05_analyze_v0_smoke.sh`，结果隔离在 `results/smoke/`；上面的 `05_analyze_v0.sh` 只汇总 controlled 最终结果。

6. 对已有 V0 输出进行 gold 事实级审计和有界敏感性分析（事实审计本身需要一张 A6000；重算前沿只使用 CPU）：

   ```bash
   source scripts/cuda_env.sh
   CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 .venv/bin/python -m src.evaluation.fact_coverage \
     --raw data/raw/hotpot_validation.parquet \
     --examples data/units/hotpot_controlled_pilot_eligible.jsonl \
     --packet-dir data/packets \
     --output results/v0_1/fact_coverage.jsonl \
     --model models/Qwen3-8B --gpu-memory-utilization 0.65 --max-model-len 2048
   bash scripts/06_analyze_v0_factaware.sh
   ```

   `06_analyze_v0_factaware.sh` 不重新运行 10,935 次 target QA，只用缓存的原回答替换事实召回并重算 fidelity、独立前沿和嵌套前沿。预注册主网格与可达断点敏感性网格分别保存，避免事后替换主实验口径。

7. 构建并分析 fact-atomic V0.2 oracle 压力测试：

   ```bash
   .venv/bin/python -m src.data.fact_atomic_pilot \
     --output data/units/hotpot_fact_atomic_candidates.jsonl
   # full-context、packet 与 exact-search 的 GPU 阶段见实验日志；已有缓存可直接重算：
   bash scripts/07_analyze_v0_2.sh
   ```

   `--levels auto` 从 exact-search 结果中枚举并合并所有数值等价的非零 fidelity 断点；它只用于敏感性分析，默认命令仍使用预注册主网格。

完整链路可通过一个可续跑入口执行：

```bash
CUDA_VISIBLE_DEVICES=<physical_gpu> bash scripts/09_run_v0_2.sh all
```

也可将最后的 `all` 换成 `prepare`、`baseline`、`packets`、`target`、`analyze` 或 `verify` 单独执行。GPU 阶段先验证缓存再加载模型；完整缓存下不会启动 vLLM。最终 `results/v0_2/verification.json` 必须同时满足 `complete=true`、`errors=[]` 和 `stage=v0_2_complete_stop_before_v1`。V1 前的交接约束见 `V1_READINESS.md`。

V0.3 的可恢复单入口为：

```bash
CUDA_VISIBLE_DEVICES=<physical_gpu> bash scripts/12_run_v0_3.sh all
```

同样支持 `prepare|baseline|packets|target|analyze|verify`。最终
`results/v0_3/gate_result.json` 为 `complete=true`、`errors=[]`，但
`scientific_gate_passed=false` 和 `decision=NO_GO_STOP_BEFORE_V1`。

Retrieval R1 与 V0.4 的可恢复入口为：

```bash
CUDA_VISIBLE_DEVICES=<physical_gpu> bash scripts/14_run_retrieval_r1.sh all
CUDA_VISIBLE_DEVICES=<physical_gpu> bash scripts/17_run_v0_4.sh all
```

R1 只有 `select` 阶段需要 Qwen3-14B；V0.4 的 `baseline|packets|target`
为 GPU 阶段。完整缓存存在时，两条 `all` 链都会先验证缓存且不会重新推理。
V0.4 当时的判定记录为 `results/v0_4/gate_result.json`。

V0.5 的最终可恢复入口为：

```bash
CUDA_VISIBLE_DEVICES=<physical_gpu> bash scripts/19_run_v0_5.sh all
```

最终权威判定已经更新为 `results/v0_5/gate_result.json`。完整缓存下会
验证 170 条 baseline、30 个 packet 文件和 21,870 个状态，而不重新加载模型。

输出对应计划的 V0 Definition of Done：`results/full_context.jsonl`、`data/packets/*.jsonl`、`results/exact_search/*.jsonl`、三个 frontier/gap CSV、`results/v0_summary.json`，以及 `results/figures/*.pdf`。

## 自动 GPU 监控

`scripts/run_v0_when_gpu_available.sh` 每 30 秒检查 CUDA 实测健康的物理 GPU 0/1/2/3（GPU 5 仅 NVML 可见、CUDA 不可用，GPU 4 的 NVML handle 异常，二者均排除）。它优先等待 Qwen3-14B teacher 单卡至少 36,000 MiB、Qwen3-8B target 单卡至少 34,000 MiB。teacher 单卡不足时，仅使用同 NUMA、PIX 互连的 GPU 0+1；两卡各至少 14,500 MiB 后，以单进程 Transformers `device_map` 分层加载 BF16 14B 模型。监控器按物理卡实时空闲量排序，并给较空闲卡最多 18 GiB、另一卡最多 11.5 GiB，使权重全部驻留 GPU；仍支持 CPU 作为安全回退。此路径不使用 NCCL，因为故障 GPU 4 会导致 NCCL 在枚举全机 NVML 拓扑时失败。target exact-search 暂不使用双卡回退，继续等待一张卡满足单卡门槛。候选设备在启动前复查 10 秒稳定性。流水线会验证并复用已完成的样本级缓存；显存竞争导致的启动失败会重新排队，packet 校验或代码错误则停止并保留日志。后端、设备选择和显存配额都会写入阶段日志及实验元数据。

自动监控已完成 smoke packet → smoke exact/analyze → controlled packet → controlled exact/analyze，并停在 V0 人工科学门槛。历史状态文件是 `logs/v0_autorun/status.tsv`，阶段日志和错误日志保存在同一目录；当前没有实验自有的模型进程占用 GPU，也不会自动开始 QLoRA/V1。

## 无模型管线检查

以下命令不下载模型、不使用真实数据，只验证 243 个状态能贯穿 representation → batched target → metrics → frontier → nested DP。结果写入 `results/dry_run/`，并明确标记为非科研结果：

```bash
bash scripts/00_dry_run.sh
python -m unittest discover -s tests -v
```

## V0 科学门槛

在进入 V1 前必须人工检查：是否存在清晰 rate–fidelity tradeoff、`G_struct` 是否足够小、高 fidelity 是否复用低 fidelity 信息、supporting facts 是否渐进进入。R1 已解决无 gold top-six 检索，V0.5 也解决了监督样本可答性，但固定五档上的 tradeoff 与事实渐进增益仍未达标。当前正式决定为最终 NO-GO，QLoRA/V1 不得启动；下一步必须是显式重新定义研究方法，而非继续编号式修补 V0。
