# Fidelity-Constrained Successive Semantic Compression
## Codex-Readable Implementation Plan

This file is the implementation blueprint for the project:

**Fidelity-Constrained Successive Semantic Compression for LLM Contexts**

Primary research goal:

> Learn a query-conditioned, fidelity-indexed, additive semantic representation for LLM contexts such that higher fidelity requirements reveal more semantic information without discarding previously revealed information, and evaluate whether LLM contexts are successively refinable.

---

# 1. Core Research Question

Traditional context compression studies:

```text
Given compression rate -> maximize fidelity
```

This project studies:

```text
Given fidelity requirement -> minimize semantic rate
```

The learned representation must support multiple fidelity operating points using one shared additive structure:

```text
R(c1) ⊆ R(c2), when c1 < c2
```

The central scientific question is:

```text
Can LLM context representations be made successively refinable with only a small gap relative to independently optimized representations?
```

---

# 2. Main Research Objects

## 2.1 Semantic Units

Split each context into non-overlapping semantic units:

```text
D = {x1, x2, ..., xn}
```

Default unit size:

```text
target = ~192 tokens
max = 256 tokens
```

Do not use overlapping chunks in the first version.

---

## 2.2 Additive Semantic Packets

Each semantic unit `xi` has two additive packets:

```text
gist_i
residual_i
```

Meaning:

```text
state 0 -> DROP
state 1 -> GIST
state 2 -> GIST + RESIDUAL
```

Important:

```text
GIST + RESIDUAL must add information.
RESIDUAL must not replace GIST.
```

This guarantees representation-level nesting.

---

## 2.3 Fidelity Thresholds

For each unit, the compressor predicts:

```text
tau_g[i] = fidelity level at which gist becomes visible
tau_r[i] = fidelity level at which residual becomes visible
```

Constraint:

```text
0 <= tau_g[i] <= tau_r[i] <= 1
```

State at fidelity level `c`:

```python
if c < tau_g[i]:
    state = 0
elif c < tau_r[i]:
    state = 1
else:
    state = 2
```

Therefore state transitions are always:

```text
0 -> 1 -> 2
```

never backward.

---

# 3. Models

## 3.1 Primary Compressor

```text
Qwen3-1.7B
```

Role:

- predict fidelity reveal thresholds
- generate semantic packets
- main trainable model

Training:

```text
LoRA / QLoRA
```

---

## 3.2 Capacity Ablation Compressor

```text
Qwen3-4B-Instruct-2507
```

Optional.

Use only after the main 1.7B system works.

Purpose:

```text
Compressor capacity vs rate-fidelity performance
```

---

## 3.3 Primary Target LLM

```text
Qwen3-8B
```

Frozen.

Role:

- full-context benchmark
- compressed-context QA
- counterfactual evaluation
- final downstream target

Use fixed prompting and decoding for all methods.

Recommended first setting:

```text
enable_thinking = False
do_sample = False
max_new_tokens = 64
```

---

## 3.4 Teacher

```text
Qwen3-14B
```

Frozen.

Offline only.

Role:

```text
semantic unit -> gist + residual candidates
```

Teacher must NOT decide:

- fidelity labels
- DROP decisions
- thresholds
- correctness
- contract satisfaction

Teacher proposes; Target + Gold data verify.

---

## 3.5 Cross-Family Target

Optional:

```text
Gemma 3 4B IT
```

Use only after the main pipeline works.

Purpose:

```text
cross-model representation transfer
```

---

# 4. Datasets

Main datasets:

```text
HotpotQA
2WikiMultiHopQA
Qasper
```

OOD / long-context extension:

```text
selected LongBench tasks
```

Suggested use:

```text
Pilot: HotpotQA
Main training: HotpotQA + 2Wiki + Qasper
OOD: LongBench
```

---

# 5. Repository Structure

```text
fidelity-compression/
│
├── configs/
│   ├── target_qwen3_8b.yaml
│   ├── teacher_qwen3_14b.yaml
│   ├── compressor_qwen3_1p7b.yaml
│   └── experiment.yaml
│
├── data/
│   ├── raw/
│   ├── normalized/
│   ├── units/
│   ├── packets/
│   ├── exact_search/
│   ├── trajectories/
│   ├── threshold_labels/
│   └── calibration/
│
├── src/
│   ├── data/
│   │   ├── normalize_hotpot.py
│   │   ├── normalize_2wiki.py
│   │   ├── normalize_qasper.py
│   │   ├── segment.py
│   │   └── schemas.py
│   │
│   ├── teacher/
│   │   ├── packet_generator.py
│   │   └── packet_validator.py
│   │
│   ├── target/
│   │   ├── qwen_runner.py
│   │   └── answer_parser.py
│   │
│   ├── representation/
│   │   ├── packet_store.py
│   │   ├── state_builder.py
│   │   └── token_counter.py
│   │
│   ├── search/
│   │   ├── exact_search.py
│   │   ├── exact_frontier.py
│   │   ├── best_nested_chain.py
│   │   └── greedy_trajectory.py
│   │
│   ├── model/
│   │   ├── fidelity_compressor.py
│   │   ├── threshold_head.py
│   │   └── input_builder.py
│   │
│   ├── train/
│   │   ├── rewrite_dataset.py
│   │   ├── threshold_dataset.py
│   │   ├── train_rewriter.py
│   │   ├── train_threshold.py
│   │   └── train_joint.py
│   │
│   ├── calibration/
│   │   ├── violation.py
│   │   └── calibrate.py
│   │
│   └── evaluation/
│       ├── qa_metrics.py
│       ├── fidelity_metrics.py
│       ├── frontier_metrics.py
│       ├── efficiency.py
│       └── aggregate.py
│
├── scripts/
│   ├── 01_prepare_data.sh
│   ├── 02_run_full_context.sh
│   ├── 03_generate_packets.sh
│   ├── 04_exact_pilot.sh
│   ├── 05_build_trajectories.sh
│   ├── 06_train_rewriter.sh
│   ├── 07_train_threshold.sh
│   ├── 08_calibrate.sh
│   └── 09_evaluate.sh
│
├── results/
├── checkpoints/
├── logs/
├── requirements.txt
└── README.md
```

---

# 6. Environment

Recommended:

```text
Python 3.11
PyTorch
transformers
datasets
accelerate
peft
bitsandbytes
vllm
sentence-transformers
scikit-learn
scipy
pandas
numpy
evaluate
rouge-score
matplotlib
tqdm
pyyaml
```

Use:

```text
Transformers + PEFT + Accelerate
```

for compressor training.

Use:

```text
vLLM
```

for large-scale Target and Teacher inference.

Do not add DeepSpeed initially.

---

# 7. Unified Data Schema

## 7.1 QA Example

```json
{
  "example_id": "hotpot_0001",
  "dataset": "hotpotqa",
  "question": "Which city ...?",
  "answer": "Stockholm",
  "context": "...",
  "units": [
    {
      "unit_id": 0,
      "text": "...",
      "supporting": false
    },
    {
      "unit_id": 1,
      "text": "...",
      "supporting": true
    }
  ]
}
```

---

## 7.2 Semantic Packet

```json
{
  "example_id": "hotpot_0001",
  "unit_id": 3,
  "source": "The company reported revenue of ...",
  "gist": "The company reported $17.36B revenue in 2025.",
  "residual": "Revenue increased 12.4%, mainly because of Asian demand.",
  "source_tokens": 125,
  "gist_tokens": 24,
  "residual_tokens": 18
}
```

---

## 7.3 Threshold Label

```json
{
  "example_id": "hotpot_0001",
  "question": "...",
  "units": [
    {
      "unit_id": 0,
      "text": "...",
      "tau_g": 0.85,
      "tau_r": 0.95
    },
    {
      "unit_id": 1,
      "text": "...",
      "tau_g": 0.60,
      "tau_r": 0.75
    }
  ]
}
```

---

# 8. Core Python Data Classes

Implement in:

```text
src/data/schemas.py
```

Suggested:

```python
from dataclasses import dataclass

@dataclass
class SemanticUnit:
    unit_id: int
    text: str
    supporting: bool = False

@dataclass
class QAExample:
    example_id: str
    dataset: str
    question: str
    answer: str
    context: str
    units: list[SemanticUnit]

@dataclass
class SemanticPacket:
    example_id: str
    unit_id: int
    source: str
    gist: str
    residual: str
    source_tokens: int
    gist_tokens: int
    residual_tokens: int
```

---

# 9. Semantic Segmentation

Implement:

```python
def segment_document(
    text: str,
    tokenizer,
    target_tokens: int = 192,
    max_tokens: int = 256
) -> list[str]:
    ...
```

Algorithm:

```text
paragraph split
-> sentence split
-> sequential merge
-> stop near target_tokens
-> hard stop at max_tokens
```

No overlap in V0/V1.

---

# 10. Packet Generation

Implement:

```text
src/teacher/packet_generator.py
```

Teacher input:

```text
one semantic unit only
```

Do NOT give the Teacher the query in the first version.

Prompt requirements:

```text
GIST:
- preserve essential facts
- preserve entities
- preserve important numbers/dates
- preserve negation
- target 20-35% source length

RESIDUAL:
- add important information missing from GIST
- do not repeat GIST
- do not introduce external information
```

Output JSON only:

```json
{
  "gist": "...",
  "residual": "..."
}
```

Retry malformed JSON at most 2-3 times.

---

# 11. Packet Validation

Implement:

```text
src/teacher/packet_validator.py
```

Checks:

```text
gist shorter than source
residual non-empty when useful
gist/residual not excessively repetitive
numbers not hallucinated
basic entity consistency
JSON valid
```

Recommended first length constraint:

```text
gist_tokens / source_tokens <= 0.45
```

Reject or regenerate badly formed packets.

---

# 12. Representation Builder

Implement:

```text
src/representation/state_builder.py
```

Core:

```python
def build_representation(
    packets,
    states
) -> str:

    output = []

    for packet, state in zip(packets, states):

        if state == 0:
            continue

        if state >= 1:
            output.append(packet.gist)

        if state >= 2:
            output.append(packet.residual)

    return "\n".join(output)
```

Always preserve original semantic-unit order.

Do NOT reorder by query relevance in the first paper.

---

# 13. Target LLM Runner

Implement:

```text
src/target/qwen_runner.py
```

Required interface:

```python
class TargetRunner:

    def __init__(self, model_name):
        ...

    def answer(
        self,
        question: str,
        context: str
    ) -> str:
        ...
```

Recommended QA prompt:

```text
Use only the context below to answer the question.
Return a concise final answer.

Context:
{context}

Question:
{question}
```

Recommended first decoding:

```text
enable_thinking = False
do_sample = False
max_new_tokens = 64
```

Keep exactly the same Target config for:

```text
full context
ours
all baselines
```

---

# 14. Full-Context Benchmark

Run before training anything.

Store:

```json
{
  "example_id": "...",
  "prediction": "...",
  "gold": "...",
  "em": 1,
  "f1": 1.0,
  "context_tokens": 4261
}
```

For compressor supervision, prefer examples where:

```text
full-context F1 >= 0.8
```

or exact match is correct.

Final test reporting must still include the full test set.

---

# 15. Fidelity Metrics

## 15.1 Answer Fidelity

For HotpotQA / 2Wiki:

```text
official EM
official F1
```

Define:

```text
F_A(R) = answer F1
```

---

## 15.2 Critical Fact Fidelity

Use supporting facts as a first proxy.

If a supporting source unit is represented by at least its gist, count it as represented.

Define:

```text
F_F(R) = supporting fact recall
```

This proxy is acceptable for V0/Pilot.

For the final paper, manually or automatically audit a subset to verify that "gist visible" actually preserves the supporting fact.

---

## 15.3 Composite Fidelity

First version:

```text
F(R) = min(F_A(R), F_F(R))
```

This avoids arbitrary weighting.

---

# 16. V0: Controlled Exact Pilot

Do this before training the compressor.

Start with:

```text
10-query smoke test
```

Then:

```text
30-query controlled pilot
```

Each query:

```text
N = 6 semantic units
```

Selection rule:

```text
include all gold supporting units
fill remaining positions with top query-relevant distractors
```

Each unit has 3 states:

```text
0, 1, 2
```

Total states:

```text
3^6 = 729
```

---

# 17. Exact Search Enumeration

Implement:

```text
src/search/exact_search.py
```

Core:

```python
from itertools import product

def enumerate_states(n_units):
    return product(range(3), repeat=n_units)
```

For each state:

```python
context = build_representation(packets, state)

prediction = target.answer(
    question,
    context
)

answer_f1 = ...
fact_recall = ...
fidelity = min(answer_f1, fact_recall)
tokens = ...
```

Store every state.

---

# 18. Batch the Exact Search

Do not call `generate()` 729 times sequentially.

Build all prompts:

```python
requests = []

for state in enumerate_states(6):

    context = build_representation(
        packets,
        state
    )

    requests.append({
        "state": state,
        "prompt": build_prompt(...)
    })
```

Then send them through vLLM batching.

This is mandatory for reasonable runtime.

---

# 19. Exact Search Output

Store:

```json
{
  "example_id": "hotpot_0001",
  "state": [0, 1, 2, 0, 1, 0],
  "tokens": 318,
  "prediction": "...",
  "answer_em": 1,
  "answer_f1": 0.95,
  "fact_recall": 0.5,
  "fidelity": 0.5
}
```

---

# 20. Independent Exact Frontier

Use fidelity levels:

```python
C_GRID = [
    0.60,
    0.70,
    0.80,
    0.90,
    0.95
]
```

For every `c`:

```python
def independent_optimum(results, c):
    feasible = [
        r for r in results
        if r["fidelity"] >= c
    ]

    if not feasible:
        return None

    return min(
        feasible,
        key=lambda x: x["tokens"]
    )
```

This gives:

```text
Rate*_independent(c)
```

---

# 21. Best Nested Chain

Implement:

```text
src/search/best_nested_chain.py
```

Partial order:

```python
def is_nested(s1, s2):
    return all(a <= b for a, b in zip(s1, s2))
```

A valid refinement path satisfies:

```text
s1 <= s2 <= ... <= sk
```

component-wise.

Use dynamic programming over fidelity levels.

Goal:

```text
minimize cumulative rate
```

subject to:

```text
F(state_k) >= c_k
state_(k-1) <= state_k
```

This yields:

```text
Rate*_nested(c)
```

---

# 22. Structural Successive Refinement Gap

Compute:

```text
G_struct(c)
=
Rate*_nested(c)
-
Rate*_independent(c)
```

Normalized:

```text
G_struct_norm(c)
=
G_struct(c)
/
Rate*_independent(c)
```

This is the most important V0 scientific metric.

Interpretation:

```text
small gap -> strong successive refinability
large gap -> nesting has a significant structural cost
```

---

# 23. V0 Go / No-Go Criteria

Do not train Qwen3-1.7B until V0 answers:

1. Is there a clear rate-fidelity tradeoff?
2. Do higher-fidelity optima largely reuse lower-fidelity information?
3. Is `G_struct(c)` reasonably small?
4. Do supporting facts progressively enter the representation?

If these are promising:

```text
continue to V1
```

If not:

```text
revisit representation design before training
```

---

# 24. Large-Scale Pseudo-Oracle

Exact search is impossible for normal long contexts.

Implement:

```text
src/search/greedy_trajectory.py
```

Start at maximum packet state:

```text
state = (2, 2, ..., 2)
```

Allow only downgrade actions:

```text
2 -> 1
1 -> 0
```

For each candidate action:

```text
delta_tokens
delta_fidelity
```

Score:

```text
score =
delta_tokens
/
(delta_fidelity + eta)
```

Suggested:

```text
eta = 0.02
```

Choose actions that save many tokens with little fidelity loss.

---

# 25. Candidate Action Pre-Filtering

Do not evaluate every possible downgrade.

Use query-unit similarity to shortlist:

```text
B = 6
```

candidate actions per greedy step.

Evaluate those in parallel with the frozen Target.

This keeps data-generation cost manageable.

---

# 26. Greedy Trajectory Output

Store nested states:

```text
S0
S1
S2
...
SK
```

from high information to low information.

Reverse them to obtain:

```text
low fidelity -> high fidelity
```

reveal trajectories.

---

# 27. Threshold Labels

Use fidelity anchors:

```python
FIDELITY_LEVELS = [
    0.60,
    0.70,
    0.80,
    0.90,
    0.95
]
```

For each semantic unit:

```text
tau_g = first fidelity level where gist is required
tau_r = first fidelity level where residual is required
```

Must satisfy:

```text
tau_g <= tau_r
```

---

# 28. Trainable Compressor Architecture

Use:

```text
Qwen3-1.7B backbone
+
two tiny scalar threshold heads
```

Do not train a second threshold model.

---

# 29. Input Builder

Implement:

```text
src/model/input_builder.py
```

Avoid special unit tokens initially.

Tokenize each section separately and record exact token spans.

Example:

```python
ids = []
unit_spans = []

question_ids = tokenizer(
    question_prefix,
    add_special_tokens=False
).input_ids

ids.extend(question_ids)

for unit in units:

    unit_ids = tokenizer(
        unit.text,
        add_special_tokens=False
    ).input_ids

    start = len(ids)

    ids.extend(unit_ids)

    end = len(ids)

    unit_spans.append((start, end))
```

This makes unit spans exact.

---

# 30. Unit Representation

Run Qwen with:

```text
output_hidden_states = True
use_cache = False
```

Take final hidden layer.

For each unit span:

```python
h_i = H[:, start:end, :].mean(dim=1)
```

This gives one vector per semantic unit.

---

# 31. Threshold Heads

Implement:

```python
a = sigmoid(gist_head(h_i))
b = sigmoid(residual_head(h_i))

tau_g = a
tau_r = a + (1 - a) * b
```

Therefore:

```text
0 <= tau_g <= tau_r <= 1
```

by construction.

This is a core method property.

---

# 32. FidelityCompressor Skeleton

Implement in:

```text
src/model/fidelity_compressor.py
```

Suggested structure:

```python
class FidelityCompressor(nn.Module):

    def __init__(self, lm):
        super().__init__()

        self.lm = lm
        hidden_size = lm.config.hidden_size

        self.gist_head = nn.Linear(
            hidden_size,
            1
        )

        self.residual_head = nn.Linear(
            hidden_size,
            1
        )

    def predict_thresholds(
        self,
        input_ids,
        attention_mask,
        unit_spans
    ):
        ...
```

---

# 33. Threshold Loss

Start simple.

Use:

```text
Smooth L1
```

for pseudo-oracle thresholds:

```text
L_thr =
SmoothL1(pred_tau_g, gold_tau_g)
+
SmoothL1(pred_tau_r, gold_tau_r)
```

Pseudo-labels are noisy, so Smooth L1 is preferable to strict MSE.

---

# 34. Optional Ranking Loss

Later add:

```text
if gold_tau_i < gold_tau_j:
    pred_tau_i should be lower than pred_tau_j
```

Example:

```python
loss_rank = relu(
    pred_i - pred_j + margin
)
```

Suggested:

```text
margin = 0.05
lambda_rank = 0.1
```

Do not add this until the basic regression pipeline works.

---

# 35. Packet Rewriter Training

Train the same Qwen3-1.7B backbone to reproduce Teacher packets.

Input:

```text
Compress the following semantic unit into additive semantic packets.

UNIT:
{source}
```

Target:

```text
<GIST>
...
</GIST>

<RESIDUAL>
...
</RESIDUAL>
```

Mask prompt tokens with:

```text
label = -100
```

Only compute LM loss on the assistant output.

---

# 36. Training Order

Use three stages.

## Stage A

Train packet rewriter only.

Goal:

```text
source -> gist + residual
```

## Stage B

Load Stage A LoRA.

Add threshold heads.

Train:

```text
query + full document -> tau_g / tau_r
```

## Stage C

Optional short joint fine-tuning.

Do not start with multitask joint training.

---

# 37. LoRA Configuration

Initial configuration:

```python
LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)
```

First learning-rate trial:

```text
1e-4
```

Initial epochs:

```text
2
```

Do not grid-search until the pipeline works.

---

# 38. 24 GB GPU Configuration

Use:

```text
4-bit NF4
BF16 compute
QLoRA
gradient checkpointing
gradient accumulation
```

Threshold task:

```text
micro batch = 1-2
```

Rewrite task:

```text
micro batch = 4-8
```

depending on context length.

---

# 39. Fidelity-Level Inference

Given predicted thresholds and fidelity level `c`:

```python
def state_from_threshold(
    c,
    tau_g,
    tau_r
):
    if c < tau_g:
        return 0
    if c < tau_r:
        return 1
    return 2
```

Apply to every semantic unit.

Then build representation in original unit order.

---

# 40. Representation-Level Monotonicity

The model guarantees:

```text
c1 < c2
=> state_i(c1) <= state_i(c2)
```

for every unit.

Therefore:

```text
R(c1) ⊆ R(c2)
```

by architecture.

Do not claim this guarantees downstream QA monotonicity.

---

# 41. Utility Monotonicity Violation

Measure separately.

Given quality scores:

```text
Q(c1), Q(c2), ..., Q(ck)
```

count cases where:

```text
Q(c_{i+1}) < Q(c_i)
```

Metric:

```text
UMVR =
number of adjacent quality decreases
/
number of adjacent fidelity pairs
```

This is an important analysis metric.

---

# 42. Independent Variable-Rate Baseline

Use the same:

```text
Qwen3-1.7B
LoRA rank
training data
training steps
```

but remove ordered thresholds.

Input:

```text
question + document + c
```

For each unit predict:

```text
0 / 1 / 2
```

with a 3-class classification head.

Different fidelity levels may produce unrelated states.

This is the fairest baseline for testing the value of nested successive refinement.

---

# 43. Learned Successive Refinement Gap

On the full test set compare:

```text
Rate_ours(c)
Rate_independent(c)
```

Define:

```text
G_learned(c)
=
Rate_ours(c)
-
Rate_independent(c)
```

Report together with:

```text
G_struct(c)
```

from the controlled exact subset.

---

# 44. Calibration

Use a completely separate calibration set.

Do not use it for:

```text
training
hyperparameter selection
threshold supervision
```

Scan:

```python
C_GRID = np.linspace(0, 1, 51)
```

For each `c`, calculate binary contract violation.

Example:

```text
violation =
answer degradation exceeds epsilon_A
OR
critical-fact degradation exceeds epsilon_F
```

---

# 45. First Calibration Implementation

Do not implement a complex conformal package first.

Use:

```text
finite fidelity grid
+
one-sided binomial upper confidence bound
+
Bonferroni correction
```

This is sufficient for the first implementation.

Later replace or formalize with LTT/CRC if needed.

---

# 46. Contract Selection

For user risk budget:

```text
alpha
```

and confidence:

```text
delta
```

compute risk upper bound:

```text
U(c)
```

Choose:

```text
c* = minimum c such that U(c) <= alpha
```

If no compressed level satisfies the contract:

```text
fallback to full context
```

---

# 47. Token Accounting

Use one tokenizer for all main rate comparisons:

```text
Qwen3-8B Target tokenizer
```

Reason:

The main question is:

```text
How many tokens does the Target actually read?
```

Do not compare methods using different tokenizers.

---

# 48. Core Evaluation Metrics

Report:

## Task

```text
EM
F1
```

## Fidelity

```text
Answer Fidelity
Supporting Fact Recall
Composite Fidelity
```

## Compression

```text
Original Tokens
Compressed Tokens
Compression Ratio
Retained Ratio
Gist Tokens
Residual Tokens
```

## Refinement

```text
G_struct
G_learned
UMVR
```

## Contract

```text
Requested Risk
Observed Risk
Risk Upper Bound
Fallback Rate
```

## Efficiency

```text
Compressor Prefill
Compressor Generation
Target Prefill
Target Generation
Total Latency
GPU-seconds
Peak GPU Memory
```

---

# 49. End-to-End Efficiency

Measure:

```text
T_ours =
T_compressor_prefill
+
T_packet_generation
+
T_target_prefill_compressed
+
T_target_generation
```

Compare with:

```text
T_full =
T_target_prefill_full
+
T_target_generation
```

Run context-length buckets:

```text
4K
8K
16K
32K
```

Find break-even length:

```text
L*
```

such that:

```text
T_ours(L*) < T_full(L*)
```

---

# 50. Packet Caching Policy

Separate two experiment modes.

## Representation-quality experiments

Packets may be pre-generated and cached.

Purpose:

```text
study representation quality
```

## End-to-end system experiments

Packets must be generated online.

Purpose:

```text
measure actual system cost
```

Never mix these two claims.

---

# 51. Baselines

Implement in this order:

```text
1. Full Context
2. Truncation
3. Retrieval Top-K
4. LongLLMLingua
5. COMI
6. Independent Variable-Rate Compressor
7. Ours
```

Add near-lossless compression later if reproducible.

The most important baseline is:

```text
Independent Variable-Rate Compressor
```

because it isolates the cost/value of the nested constraint.

---

# 52. Main Paper Figures

Required figures:

```text
Figure 1: Exact / learned Rate-Fidelity Frontier
Figure 2: Fidelity level c vs retained tokens / packet states
Figure 3: Structural and learned Successive Refinement Gap
Figure 4: Requested Risk vs Observed Risk
Figure 5: Context Length vs End-to-End Cost
```

---

# 53. Main Paper Tables

Recommended:

```text
Table 1: Overall Rate-Fidelity Results
Table 2: Contract Satisfaction
Table 3: Nested vs Independent
Table 4: Ablations
Table 5: Cross-Model / OOD
```

---

# 54. Development Order

Follow exactly this order.

## Phase V0-A

Implement:

```text
normalize_hotpot.py
segment.py
qwen_runner.py
```

Goal:

```text
full-context QA works
```

## Phase V0-B

Implement:

```text
packet_generator.py
packet_validator.py
state_builder.py
```

Goal:

```text
semantic packets are valid
```

## Phase V0-C

Implement:

```text
exact_search.py
exact_frontier.py
best_nested_chain.py
```

Goal:

```text
measure G_struct before training anything
```

## Phase V1-A

Implement:

```text
greedy_trajectory.py
threshold_dataset.py
```

Goal:

```text
large-scale pseudo-oracle labels
```

## Phase V1-B

Implement:

```text
fidelity_compressor.py
train_rewriter.py
train_threshold.py
```

Goal:

```text
learn fidelity-indexed representation
```

## Phase V1-C

Implement:

```text
calibrate.py
full_evaluation.py
```

Goal:

```text
contract-controlled final system
```

---

# 55. First Smoke Test

Use:

```text
10 HotpotQA examples
5 units each
3^5 = 243 states
```

Only test code correctness.

---

# 56. Controlled Pilot

Use:

```text
30 HotpotQA examples
6 units each
3^6 = 729 states
```

Total Target evaluations:

```text
30 * 729 = 21,870
```

Use vLLM batching.

If affordable, later increase to:

```text
50-100 queries
```

---

# 57. First Scientific Gate

Do not train the compressor before checking:

```text
Rate-Fidelity tradeoff exists
G_struct is reasonably small
higher fidelity substantially reuses lower-fidelity information
supporting facts enter progressively
```

If these fail, revisit the representation design.

---

# 58. Compressor Pilot Scale

After V0 succeeds:

```text
2,000 HotpotQA queries
```

Build pseudo-trajectories.

Train first Qwen3-1.7B prototype.

The first model only needs to demonstrate:

```text
c increases -> retained tokens increase
c increases -> fidelity tends to improve
nestedness holds 100%
```

Do not require it to beat COMI immediately.

---

# 59. Full Training Scale

Only after the pilot works.

Recommended total:

```text
15K-30K query trajectories
```

from:

```text
HotpotQA
2WikiMultiHopQA
Qasper
```

---

# 60. Reproducibility Requirements

Every experiment record:

```text
experiment_id
git commit hash
model name
model revision
dataset revision
LoRA config
learning rate
seed
context length
batch size
gradient accumulation
GPU type
training time
checkpoint path
generation parameters
```

Use:

```text
TensorBoard or Weights & Biases
```

---

# 61. Seeds

Development:

```text
42
```

Final core experiments:

```text
42
123
2026
```

Report:

```text
mean ± std
```

and 95% confidence intervals where appropriate.

---

# 62. Immediate Codex Task List

Codex should implement in the following order.

## Task 1

Create repository structure and configuration loader.

## Task 2

Implement unified schemas and HotpotQA normalizer.

## Task 3

Implement non-overlapping semantic segmentation.

## Task 4

Implement frozen Qwen3-8B Target runner.

## Task 5

Run and validate full-context benchmark.

## Task 6

Implement Qwen3-14B semantic packet generator.

## Task 7

Implement packet validation and caching.

## Task 8

Implement representation builder.

## Task 9

Implement exact state enumeration.

## Task 10

Implement batched vLLM exact-state inference.

## Task 11

Implement independent exact frontier extraction.

## Task 12

Implement dynamic-programming best nested chain.

## Task 13

Output `G_struct(c)` and initial plots.

## STOP POINT

Do not implement QLoRA training until V0 results are inspected.

## Task 14

Implement greedy pseudo-oracle trajectory generation.

## Task 15

Convert trajectories to threshold supervision.

## Task 16

Implement Qwen3-1.7B fidelity compressor.

## Task 17

Implement packet rewriter LoRA training.

## Task 18

Implement threshold-head training.

## Task 19

Implement independent variable-rate baseline.

## Task 20

Implement calibration.

## Task 21

Implement main evaluation suite.

## Task 22

Implement end-to-end efficiency benchmark.

---

# 63. Core APIs

Keep these stable.

## Representation

```python
R = build_representation(
    packets,
    states
)
```

## Target

```python
answer = target.answer(
    question,
    R
)
```

## Compressor

```python
tau_g, tau_r = compressor.predict_thresholds(
    question,
    units
)
```

## State selection

```python
states = states_at_fidelity(
    c,
    tau_g,
    tau_r
)
```

## Calibration

```python
c_star = calibrate_contract(
    calibration_results,
    alpha,
    delta
)
```

---

# 64. Critical Research Constraints

Codex must preserve the following design constraints unless explicitly changed:

1. Do not replace additive packets with independent summaries.
2. Do not allow higher fidelity to remove already revealed information.
3. Do not use Teacher outputs as final correctness labels.
4. Target LLM remains frozen.
5. Compressor is the main trainable model.
6. Calibration data must be separate from train/validation data.
7. Token accounting uses the Target tokenizer.
8. Exact-search pilot must be completed before large-scale training.
9. Do not add new neural modules unless a measured failure requires them.
10. End-to-end efficiency claims must include compressor cost.

---

# 65. Definition of Done for V0

V0 is complete when the repository can produce:

```text
results/full_context.jsonl
data/packets/*.jsonl
results/exact_search/*.jsonl
results/exact_frontier.csv
results/nested_frontier.csv
results/structural_gap.csv
results/figures/rate_fidelity_frontier.pdf
results/figures/structural_gap.pdf
```

and answer:

```text
Does a low-cost nested semantic refinement path exist close to the independent exact optimum?
```

---

# 66. Definition of Done for V1

V1 is complete when the system can:

```text
1. predict tau_g and tau_r for every semantic unit
2. construct R(c) for arbitrary c
3. guarantee representation-level nestedness
4. run the frozen Target once
5. calibrate user risk requirement to c*
6. report rate-fidelity and successive-refinement gaps
7. compare against independent variable-rate compression
8. measure end-to-end latency including compressor cost
```

---

# 67. Final Research Objective

The codebase should ultimately test the hypothesis:

```text
A single query-conditioned fidelity-indexed additive semantic representation
can serve multiple fidelity operating points with only a small rate penalty
relative to independently optimized representations.
```

The primary scientific quantities are:

```text
Rate(c)
Fidelity(c)
G_struct(c)
G_learned(c)
UMVR
Contract Violation Risk
End-to-End Cost
```

Do not optimize for engineering complexity.

Optimize for a clean test of the scientific hypothesis.
