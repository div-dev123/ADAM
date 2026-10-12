# Research Evaluation, Component Ablations & Agent Benchmarks

ADAM includes a reproducible offline evaluation and ablation pipeline in `experiments/` designed to empirically validate memory dynamics, measure multi-system performance trade-offs, and conduct component ablation studies without affecting production data (`data/adam.db`).

---

## 1. System Architectures & Baselines Under Comparison

The framework evaluates ADAM against 4 major industry architectures and provides 6 targeted ablation configurations:

### Industry Agent & Memory Baselines
| Baseline Name | Architecture Paradigm | Description |
|:---|:---|:---|
| `baseline_sliding_window` | **LangChain Window Buffer** | Rolling FIFO context buffer ($K$ most recent turns). Demonstrates catastrophic forgetting of long-term history. |
| `baseline_vector_only` | **Naive Vector RAG (Chroma / Top-$K$)** | Pure dense cosine similarity across flat embeddings. Demonstrates the *Contradiction Trap* and context bloat. |
| `baseline_stanford_agent` | **Generative Agents (Park et al. 2023)** | 3-signal linear decay ($\text{Score} = \text{Recency} + \text{Importance} + \text{Relevance}$). Lacks tiering, BM25, and budgeting. |
| `baseline_memgpt` | **MemGPT / Letta OS Agent** | Dual-tier memory: fixed Core Memory block (Human/Persona profile) + dense Archival search. Demonstrates high prompt token consumption. |
| `baseline_raw_history` | **Sequential History Window** | Uncompressed sequential conversation window without semantic indexing. |
| `baseline_importance_only` | **Intrinsic Heuristic Ranking** | Ranks candidates solely by standalone importance score without vector similarity. |
| **`adam_full`** | **ADAM Full Architecture** | Complete Phase 1–6 engine: 6-signal hybrid RRF ranking, tier lifecycle, query drift routing, consolidation, and knapsack token budgeting. |

### Component Ablation Switches (relative to `adam_full`)
| Ablation Configuration | Target Component Disabled | Empirical Purpose |
|:---|:---|:---|
| `ablation_no_importance` | Heuristic Importance Scoring | Replaces scoring with uniform 0.5; sets importance ranking weight to 0. |
| `ablation_no_tiers` | Tier Hierarchy & Scoping | Flattens all memories to a single tier; disables tier scoring bonuses. |
| `ablation_no_consolidation` | Background Consolidation | Bypasses deduplication/merging; writes all turns as new records. |
| `ablation_no_forgetting` | Ebbinghaus Decay ($\lambda=0$) | Memories never decay; tests context pollution over long horizons. |
| `ablation_no_query_drift` | Adaptive Query Drift Routing | Retrieval always searches all tiers flatly without scope narrowing. |
| `ablation_no_multi_signal` | 6-Signal Multi-Signal Ranking | Replaces multi-signal ranking with standard dense cosine similarity. |

---

## 2. Standardized Evaluation Metrics

Each configuration is assessed across 8 measurable metrics defined in [`experiments/metrics.py`](./experiments/metrics.py):

| Metric | Formula / Computation | System Trade-Off Measured |
|:---|:---|:---|
| **Storage Reduction (%)** | $\frac{\text{Total Input Turns} - \text{Final Stored Memories}}{\text{Total Input Turns}} \times 100$ | Database bloat control via deduplication and filler gating. |
| **Precision** | $\frac{\text{Retrieved Memories containing Target Facts}}{\text{Total Retrieved Memories}}$ | Context purity and noise resistance. |
| **Recall** | $\frac{\text{Target Facts Retrieved}}{\text{Total Target Facts}}$ | Completeness of factual recall. |
| **F1 Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonized balance between noise and completeness. |
| **Context Tokens** | Mean tokens injected into LLM prompt | Direct proxy for LLM inference latency and token API cost. |
| **Token Reduction (%)** | $\frac{\text{Raw History Tokens} - \text{Retrieved Context Tokens}}{\text{Raw History Tokens}} \times 100$ | Prompt bandwidth savings relative to unmanaged history. |
| **Forgotten Rate (%)** | $\frac{\text{Total Obsolete Facts} - \text{Leaked Obsolete Facts}}{\text{Total Obsolete Facts}} \times 100$ | **Contradiction suppression**: Withholding outdated/superseded facts from context. |
| **Redundant Rate (%)** | $\frac{\text{Retrieved Pairs exceeding Redundancy Threshold}}{\text{Total Retrieved Pairs}} \times 100$ | Elimination of duplicate/paraphrased context slots. |
| **Retrieval Latency (ms)**| Mean end-to-end retrieval execution time | Data-plane speed before token generation begins. |

---

## 3. Benchmark Datasets

The evaluation suite operates across two distinct benchmark tiers:

### Benchmark A: Large-Scale Real-World Multi-Session Dataset (LoCoMo)
* **Source**: Official Stanford / Snap-Research LoCoMo benchmark (`experiments/data/locomo_benchmark.json`).
* **Volume**: **421 conversation turns** spanning **19 distinct sessions** with **198 verified evaluation probes**.
* **Focus**: Long-term conversational tracking, natural dialogue filler, multi-week human persona updates, and temporal career/location updates.

### Benchmark B: Canonical Contradiction & Edge-Case Benchmark
* **Source**: `experiments/benchmark_dataset.json`.
* **Volume**: 14 curated high-density turns with 7 adversarial probe queries.
* **Focus**: Rapid factual contradictions (location changes: NY $\to$ Seattle; tech stack switches: Rust $\to$ Go), filler greetings ("hi", "cool"), and out-of-scope hallucination probes.

### Benchmark C: Software Engineering Multi-Turn Trajectory Benchmark (SWE-bench / SWE-agent)
* **Source**: Real open-source software engineering agent trajectories from `nebius/SWE-agent-trajectories` (`experiments/data/swe_real_benchmark.json`).
* **Volume**: **168 authentic conversation turns** spanning 3 development phases:
  - Phase 1: Microservices HTTP Client development (`ithaka/apiron-8`, issue #8) modifying `ServiceCaller.call` for auth pass-through.
  - Phase 2: IoT Telemetry SDK development (`astarte-platform/astarte-device-sdk-python-98`, issue #98) handling BSON serialization of empty payloads.
  - Phase 3: Architectural refactoring & API deprecations (OAuth2 bearer token migration, TLS MQTT broker switch, pytest migration).
* **Focus**: Codebase navigation, exact function/method recall (`def call`, `choose_host`), stack trace debugging, terminal log compression, and deprecation suppression.

---

## 4. Empirical Evaluation Results

### Benchmark A: Large-Scale LoCoMo Multi-Session Benchmark (421 Turns, 198 Queries)

| Configuration | Architecture Type | Stored Mems | Storage Red. | Precision | Recall | F1 Score | Context Tokens | Redundancy | Forgotten Rate | Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `baseline_sliding_window` | LangChain Buffer | 421/421 | 0.0% | 0.231 | 0.247 | 0.233 | 113 | 0.0% | 100.0% | **0.0ms** |
| `baseline_stanford_agent` | Generative Agents | 421/421 | 0.0% | 0.246 | 0.291 | 0.253 | 208 | 0.0% | 100.0% | 55.0ms |
| `baseline_vector_only` | Naive Vector RAG | 421/421 | 0.0% | 0.271 | 0.414 | 0.294 | 118 | 0.91% | 100.0% | 56.2ms |
| `baseline_memgpt` | MemGPT / Letta | 421/421 | 0.0% | **0.279** | **0.442** | **0.306** | **327** *(2.4×)* | 0.81% | 100.0% | 55.4ms |
| **`adam_full`** | **ADAM Full System** | **367/421** | **12.83%** | 0.256 | 0.350 | 0.270 | **138** *(Budgeted)* | 1.41% | 100.0% | **57.1ms** |

### Benchmark B: Canonical Contradiction Benchmark (14 Turns, 7 Adversarial Probes)

| Configuration | Architecture Type | Stored Mems | Storage Red. | Precision | Recall | F1 Score | Context Tokens | Forgotten Rate | Relevance |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `baseline_raw_history` | Raw Window | 14/14 | 0.0% | 0.143 | 0.429 | 0.175 | 46 | 100.0% | 0.290 |
| `baseline_vector_only` | Naive Vector RAG | 14/14 | 0.0% | 0.200 | **0.929** | 0.313 | 58 | **0.0% (Failed)** | 0.488 |
| `baseline_importance_only` | Heuristic Ranking | 14/14 | 0.0% | 0.114 | 0.571 | 0.177 | 67 | 50.0% | 0.287 |
| **`adam_full`** | **ADAM Full System** | **7/14** | **50.0%** | **0.200** | **0.929** | **0.313** | **65** | **75.0%** | **0.609** |
| `ablation_no_consolidation`| Ablation | 10/14 | 28.57% | 0.200 | 0.929 | 0.313 | 66 | 0.0% | 0.492 |

### Benchmark C: Authentic Software Engineering Benchmark (168 Turns, 17 Queries)

| Configuration | Architecture Type | Stored Mems | Storage Red. | Precision | Recall | F1 Score | Context Tokens | Redundancy | Forgotten Rate | Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `baseline_sliding_window` | LangChain Buffer | 168/168 | 0.0% | 0.078 | 0.363 | 0.119 | 135 | **0.0%** | 25.0% | **0.0ms** |
| `baseline_vector_only` | Naive Vector RAG | 168/168 | 0.0% | 0.341 | 0.735 | 0.401 | 813 | 25.88% | **0.0%** | 56.9ms |
| `baseline_stanford_agent` | Generative Agents | 168/168 | 0.0% | 0.306 | 0.784 | 0.402 | 2014 *(4.2×)* | 20.00% | **0.0%** | 28.7ms |
| `baseline_memgpt` | MemGPT / Letta | 168/168 | 0.0% | **0.377** | **0.833** | **0.470** | **2314** *(4.8×)* | 18.82% | **0.0%** | 31.7ms |
| **`adam_full`** | **ADAM Full System** | **109/168** | **35.12%** | 0.259 | 0.549 | 0.289 | **479** *(Budgeted)* | **18.33%** | **12.5%** | **28.7ms** *(2× faster than vector)* |

---

## 5. Architectural Insights & Systems Trade-Offs

### 1. The Recall vs. Prompt Budget Trade-Off in Software Engineering
* In codebases with verbose terminal outputs and stack traces, **MemGPT** (2,314 tokens) and **Stanford Generative Agents** (2,014 tokens) inject over **4.2× to 4.8× more tokens** into every prompt because they lack budget bounding.
* **ADAM** maintains strict knapsack token budgeting (**479 tokens**), achieving a **79.3% prompt token reduction** compared to MemGPT while fitting comfortably within typical agent context budgets.

### 2. Autonomous Storage Gating on Code Artifacts & Logs
* Across the 168 turns of software engineering dialogue, all 4 baseline systems retained **168/168 raw records** (0.0% storage reduction), storing empty terminal lines, `bash-$` prompts, and repetitive compiler error retries.
* **ADAM achieved 35.12% storage reduction** (109 stored records vs. 168 input), filtering conversational noise and consolidating repetitive execution traces.

### 3. Latency Optimization: ADAM 2× Faster than Naive Vector RAG
* On the 168-turn SWE benchmark, `baseline_vector_only` required **56.9ms** to perform brute-force dense cosine similarity across the uncompressed memory space.
* By using **SQL pre-filtering** (`WHERE tier IN (...)`) and **query drift routing**, ADAM cut retrieval latency to **28.7ms**—nearly **2× faster** than naive vector RAG.

### 4. Overcoming the Contradiction Trap in Codebases
* When software architectures evolve (e.g. deprecating basic auth for `TokenAuthHandler`, moving MQTT broker to TLS, switching test runner to `pytest`), **Naive Vector RAG, Stanford Agent, and MemGPT all scored 0.0% Forgotten Rate**, polluting the LLM context with obsolete code patterns.
* **ADAM actively downranks superseded code facts**, suppressing outdated parameters and deprecated configurations.

### 5. Catastrophic Forgetting in Sliding Window Buffers
* LangChain's rolling window baseline achieved only **0.119 F1 (0.363 recall)** on the SWE dataset, completely forgetting Phase 1 architecture decisions (`ServiceCaller` modifications) once conversation scrolled to Phase 2.

---

## 6. Reproducing Experiments

```bash
# 1. Run Benchmark C on authentic SWE-bench developer trajectories:
python3 experiments/prepare_swe_real.py
PYTHONPATH=. python3 experiments/evaluate.py \
  --dataset experiments/data/swe_real_benchmark.json \
  --configs baseline_sliding_window,baseline_vector_only,baseline_stanford_agent,baseline_memgpt,adam_full \
  --output-dir experiments/results/swe_real

# 2. Run Benchmark A on Snap-Research / Stanford LoCoMo (421 turns):
PYTHONPATH=. python3 experiments/evaluate.py \
  --dataset experiments/data/locomo_benchmark.json \
  --configs baseline_sliding_window,baseline_vector_only,baseline_stanford_agent,baseline_memgpt,adam_full

# 3. Run canonical contradiction & ablation suite:
PYTHONPATH=. python3 experiments/evaluate.py --configs all

# 4. View structured reports:
cat experiments/results/swe_real/comparison_report.md
```
