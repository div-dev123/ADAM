# Research Evaluation & Component Ablation Framework

ADAM includes a reproducible offline evaluation and ablation pipeline in `experiments/` designed to empirically validate memory dynamics without affecting production data (`data/adam.db`).

## 1. Available Baselines & Ablation Switches

The framework evaluates 4 canonical baselines and 6 targeted ablation configurations:

| Category | Configuration | Description |
|:---|:---|:---|
| **Baselines** | `baseline_raw_history` | Sequential conversation window without semantic indexing or filtering |
| | `baseline_vector_only` | Pure dense vector retrieval (top-$k$ cosine similarity) with uniform flat tiers |
| | `baseline_importance_only` | Ranks candidate memories solely by heuristic importance score |
| | `adam_full` | Complete ADAM architecture (Phase 1–6: drift, hybrid RRF, 6-signal ranking, context budgeting, lifecycle) |
| **Ablations** | `ablation_no_importance` | Uniform 0.5 importance scoring; zero importance ranking weight |
| | `ablation_no_tiers` | Single flat tier; zero tier score bonus; disabled scope filtering |
| | `ablation_no_consolidation` | Bypasses consolidation; writes all items directly as new without duplicate/contradiction resolution |
| | `ablation_no_forgetting` | Disables Ebbinghaus decay ($\lambda=0$); memories are never pruned |
| | `ablation_no_query_drift` | Always queries all tiers flatly without scope narrowing |
| | `ablation_no_multi_signal` | Replaces multi-signal ranking with standard dense cosine similarity |

## 2. Measurable Metrics

Each configuration is assessed across 7 standardized metrics:
- **Retrieval Precision & Recall**: Ground-truth fact retrieval accuracy across evaluation queries.
- **Memory Count & Storage Reduction (%)**: Percentage reduction in stored memories compared to raw turns.
- **Context Size & Token Reduction (%)**: Prompt token savings relative to stuffing the entire raw conversation history.
- **Forgotten-Memory Rate (%)**: Suppression of obsolete/superseded facts (e.g. outdated location or tech stack choices) to prevent hallucinated contradiction in prompt context.
- **Redundant Memory Rate (%)**: Percentage of retrieved context pairs exceeding semantic redundancy threshold.
- **Context Relevance Score**: Harmonized alignment of retrieved context with query intent.
- **Retrieval Latency (ms)**: End-to-end retrieval latency per query (mean and $p95$).

## 3. Canonical Benchmark Dataset

Located at `experiments/benchmark_dataset.json`, the default benchmark includes:
- **14 conversation turns**: Essential technical facts, repeated facts, filler greetings/acknowledgments, and contradictory updates (e.g., location move from New York to Seattle, language switch from Rust to Go).
- **7 probe queries**: Evaluating contradiction resolution, precise configuration recall, preference retrieval, and hallucination control on out-of-scope queries.
- **Modularity**: Custom JSON datasets matching the `BenchmarkDataset` schema can be evaluated seamlessly via `--dataset <path>`.

## 4. Running Experiments

```bash
# Run all baselines and ablations on the canonical benchmark
python experiments/evaluate.py --configs all

# Run specific configurations
python experiments/evaluate.py --configs adam_full,baseline_vector_only,ablation_no_consolidation

# Run on a custom dataset with custom budget
python experiments/evaluate.py --dataset path/to/dataset.json --token-budget 800

# View structured results
cat experiments/results/comparison_report.md
```

## 5. Empirical Results

Benchmark results on the canonical 14-turn dataset:

| Configuration | Type | Stored | Storage Red. | Precision | Recall | F1 | Context Tokens | Forgotten Rate | Relevance |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `baseline_raw_history` | Baseline | 14/14 | 0.0% | 0.143 | 0.429 | 0.175 | 46 | 100.0% | 0.290 |
| `baseline_vector_only` | Baseline | 14/14 | 0.0% | 0.200 | 0.929 | 0.313 | 58 | 0.0% | 0.488 |
| `baseline_importance_only` | Baseline | 14/14 | 0.0% | 0.114 | 0.571 | 0.177 | 67 | 50.0% | 0.287 |
| `adam_full` | Full System | **7/14** | **50.0%** | **0.200** | **0.929** | **0.313** | **65** | **75.0%** | **0.609** |
| `ablation_no_consolidation` | Ablation | 10/14 | 28.57% | 0.200 | 0.929 | 0.313 | 66 | 0.0% | 0.492 |

*Results are automatically exported to `experiments/results/results.json`, `experiments/results/results.csv`, and `experiments/results/comparison_report.md`.*
