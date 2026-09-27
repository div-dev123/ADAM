"""ADAM Research Evaluation & Ablation Pipeline CLI."""

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Dict, List, Optional

from app.retrieval.embeddings import EmbeddingService
from experiments.configs import ExperimentConfig, get_standard_configurations
from experiments.dataset import BenchmarkDataset, get_default_benchmark_dataset
from experiments.metrics import (
    AggregateExperimentMetrics,
    QueryMetricResult,
    aggregate_metrics,
    compute_query_metrics,
    estimate_tokens,
)
from experiments.runners import create_runner


def run_single_experiment(
    config: ExperimentConfig,
    dataset: BenchmarkDataset,
    embeddings: EmbeddingService,
    work_dir: Path,
    verbose: bool = False,
) -> AggregateExperimentMetrics:
    """Executes a single baseline or ablation configuration on the evaluation dataset."""
    runner = create_runner(config, embeddings, work_dir)
    try:
        # 1. Ingestion Phase
        start_ingest = time.perf_counter()
        stored_count = runner.ingest(dataset.user_id, dataset.turns)
        ingest_time = time.perf_counter() - start_ingest

        if verbose:
            print(f"[{config.name}] Ingested {len(dataset.turns)} turns -> {stored_count} stored memories in {ingest_time:.2f}s")

        # Compute raw history total tokens for normalization
        raw_history_tokens = sum(estimate_tokens(t.content) for t in dataset.turns)

        # 2. Retrieval Evaluation Phase
        query_results: List[QueryMetricResult] = []
        for q in dataset.queries:
            t0 = time.perf_counter()
            retrieved = runner.retrieve(dataset.user_id, q.query)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            q_metric = compute_query_metrics(
                query_id=q.query_id,
                query_text=q.query,
                retrieved_contents=retrieved,
                target_facts=q.target_facts,
                obsolete_facts=q.obsolete_facts,
                latency_ms=latency_ms,
                redundancy_threshold=config.redundancy_threshold,
            )
            query_results.append(q_metric)

        # 3. Aggregate Metrics Computation
        agg = aggregate_metrics(
            config_name=config.name,
            description=config.description,
            is_baseline=config.is_baseline,
            ablation_target=config.ablation_target,
            total_input_turns=len(dataset.turns),
            final_stored_memories=stored_count,
            raw_history_total_tokens=raw_history_tokens,
            query_results=query_results,
        )
        return agg
    finally:
        runner.cleanup()


def generate_markdown_report(
    results: List[AggregateExperimentMetrics],
    dataset: BenchmarkDataset,
    output_path: Path,
) -> str:
    """Generates an extensive markdown analysis and ablation report."""
    md = []
    md.append(f"# ADAM Research Evaluation & Component Ablation Report\n")
    md.append(f"**Dataset**: {dataset.name} ({len(dataset.turns)} turns, {len(dataset.queries)} evaluation queries)\n")
    md.append(f"**Description**: {dataset.description}\n")
    md.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n\n")

    md.append("## 1. Executive Summary Table\n\n")
    md.append("| Configuration | Type | Stored Mems | Storage Red. | Precision | Recall | F1 Score | Context Tokens | Token Red. | Latency | Redundancy | Forgotten Rate | Relevance |")
    md.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for r in results:
        cfg_type = "Baseline" if r.is_baseline else ("Ablation" if r.ablation_target else "Full System")
        md.append(
            f"| `{r.config_name}` | {cfg_type} | {r.final_stored_memories}/{r.total_input_turns} | "
            f"{r.storage_reduction_pct}% | {r.mean_precision:.3f} | {r.mean_recall:.3f} | {r.mean_f1:.3f} | "
            f"{r.avg_context_tokens:.0f} | {r.context_token_reduction_pct}% | {r.mean_latency_ms:.1f}ms | "
            f"{r.redundant_memory_rate}% | {r.forgotten_memory_rate}% | {r.context_relevance_score:.3f} |"
        )

    md.append("\n\n## 2. Key Insights & Ablation Findings\n\n")

    # Find ADAM full and compare against baselines and ablations
    adam_full = next((r for r in results if r.config_name == "adam_full"), None)
    vector_only = next((r for r in results if r.config_name == "baseline_vector_only"), None)
    raw_hist = next((r for r in results if r.config_name == "baseline_raw_history"), None)
    no_consol = next((r for r in results if r.config_name == "ablation_no_consolidation"), None)
    no_drift = next((r for r in results if r.config_name == "ablation_no_query_drift"), None)
    no_ms = next((r for r in results if r.config_name == "ablation_no_multi_signal"), None)

    if adam_full and vector_only:
        md.append(f"### ADAM Full vs. Dense Vector Retrieval Baseline\n")
        f1_diff = ((adam_full.mean_f1 - vector_only.mean_f1) / max(0.001, vector_only.mean_f1)) * 100.0
        tok_diff = adam_full.context_token_reduction_pct - vector_only.context_token_reduction_pct
        md.append(f"- **Retrieval F1 Score**: ADAM Full ({adam_full.mean_f1:.3f}) vs Vector Only ({vector_only.mean_f1:.3f}) — **{f1_diff:+.1f}%** relative improvement.\n")
        md.append(f"- **Context Efficiency**: Context token reduction improved from {vector_only.context_token_reduction_pct}% to {adam_full.context_token_reduction_pct}%, conserving token budget for the LLM.\n")
        md.append(f"- **Forgotten / Obsolete Fact Suppression**: ADAM Full achieves **{adam_full.forgotten_memory_rate}%** suppression of outdated/superseded memories (e.g., location and programming language updates), preventing hallucinated contradictory context.\n\n")

    if adam_full and no_consol:
        md.append(f"### Impact of Memory Consolidation (`ablation_no_consolidation`)\n")
        md.append(f"- Without consolidation, memory storage grew from **{adam_full.final_stored_memories}** to **{no_consol.final_stored_memories}** items ({no_consol.storage_reduction_pct}% vs {adam_full.storage_reduction_pct}% reduction).\n")
        md.append(f"- Redundancy in retrieved context rose to **{no_consol.redundant_memory_rate}%** due to unmerged repeated facts.\n\n")

    if adam_full and no_ms:
        md.append(f"### Impact of Multi-Signal Ranking (`ablation_no_multi_signal`)\n")
        md.append(f"- Disabling multi-signal ranking reduces precision from **{adam_full.mean_precision:.3f}** to **{no_ms.mean_precision:.3f}** because recency, importance, and BM25 lexical alignment are ignored.\n\n")

    md.append("## 3. Metric Definitions\n\n")
    md.append("- **Storage Reduction (%)**: `(1 - final_memories / total_input_turns) * 100`. Measures deduplication and filler filtering.\n")
    md.append("- **Precision**: Fraction of retrieved memories containing true target facts.\n")
    md.append("- **Recall**: Fraction of ground truth facts successfully retrieved across evaluation queries.\n")
    md.append("- **Context Token Reduction (%)**: `(1 - avg_retrieved_tokens / raw_history_tokens) * 100`. Demonstrates prompt savings.\n")
    md.append("- **Redundant Memory Rate (%)**: Percentage of retrieved context pairs exceeding semantic redundancy threshold.\n")
    md.append("- **Forgotten Memory Rate (%)**: Percentage of obsolete/contradicted facts successfully withheld from retrieval context.\n")
    md.append("- **Retrieval Latency (ms)**: End-to-end retrieval latency in milliseconds.\n")

    content = "\n".join(md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return content


def save_csv_results(results: List[AggregateExperimentMetrics], output_path: Path) -> None:
    """Exports summary tabular metrics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_name",
        "type",
        "description",
        "stored_memories",
        "total_turns",
        "storage_reduction_pct",
        "mean_precision",
        "mean_recall",
        "mean_f1",
        "avg_context_tokens",
        "context_token_reduction_pct",
        "mean_latency_ms",
        "p95_latency_ms",
        "redundant_memory_rate",
        "forgotten_memory_rate",
        "context_relevance_score",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            cfg_type = "Baseline" if r.is_baseline else ("Ablation" if r.ablation_target else "Full System")
            writer.writerow({
                "config_name": r.config_name,
                "type": cfg_type,
                "description": r.description,
                "stored_memories": r.final_stored_memories,
                "total_turns": r.total_input_turns,
                "storage_reduction_pct": r.storage_reduction_pct,
                "mean_precision": r.mean_precision,
                "mean_recall": r.mean_recall,
                "mean_f1": r.mean_f1,
                "avg_context_tokens": r.avg_context_tokens,
                "context_token_reduction_pct": r.context_token_reduction_pct,
                "mean_latency_ms": r.mean_latency_ms,
                "p95_latency_ms": r.p95_latency_ms,
                "redundant_memory_rate": r.redundant_memory_rate,
                "forgotten_memory_rate": r.forgotten_memory_rate,
                "context_relevance_score": r.context_relevance_score,
            })


def save_json_results(results: List[AggregateExperimentMetrics], output_path: Path) -> None:
    """Exports structured full JSON containing per-query breakdowns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_experiments": len(results),
        "experiments": [r.to_dict() for r in results],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="ADAM Research Evaluation & Component Ablation Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default="experiments/benchmark_dataset.json",
        help="Path to benchmark dataset JSON file",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="all",
        help="Comma-separated list of configuration names to run, or 'all'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Directory to store evaluation results and reports",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name for dense embeddings",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging during execution",
    )

    args = parser.parse_args()

    # Load dataset
    dataset_path = Path(args.dataset)
    if dataset_path.exists():
        dataset = BenchmarkDataset.load_json(dataset_path)
    else:
        print(f"Dataset {dataset_path} not found. Generating default benchmark dataset...")
        dataset = get_default_benchmark_dataset()
        dataset.save_json(dataset_path)

    print(f"Loaded benchmark dataset: '{dataset.name}' ({len(dataset.turns)} turns, {len(dataset.queries)} queries)")

    # Resolve configs
    all_configs = get_standard_configurations()
    if args.configs.lower() == "all":
        selected_configs = list(all_configs.values())
    else:
        names = [n.strip() for n in args.configs.split(",")]
        selected_configs = [all_configs[n] for n in names if n in all_configs]
        if not selected_configs:
            print(f"Error: None of requested configs {names} found in registry.")
            return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp_stores"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing embedding service ({args.model})...")
    embeddings = EmbeddingService(args.model)

    results: List[AggregateExperimentMetrics] = []
    print(f"\nRunning evaluation on {len(selected_configs)} configurations...")
    print("=" * 95)
    print(f"{'Configuration':<30} | {'Stored':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<6} | {'Tokens':<6} | {'Suppr.%':<7}")
    print("-" * 95)

    for cfg in selected_configs:
        res = run_single_experiment(cfg, dataset, embeddings, temp_dir, verbose=args.verbose)
        results.append(res)
        print(
            f"{res.config_name:<30} | {f'{res.final_stored_memories}/{res.total_input_turns}':<8} | "
            f"{res.mean_precision:<9.3f} | {res.mean_recall:<8.3f} | {res.mean_f1:<6.3f} | "
            f"{int(res.avg_context_tokens):<6} | {f'{res.forgotten_memory_rate}%':<7}"
        )

    print("=" * 95)

    # Save artifacts
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    report_path = output_dir / "comparison_report.md"

    save_json_results(results, json_path)
    save_csv_results(results, csv_path)
    generate_markdown_report(results, dataset, report_path)

    # Clean up temp_dir
    try:
        for f in temp_dir.glob("*.db*"):
            f.unlink()
        temp_dir.rmdir()
    except Exception:
        pass

    print(f"\nEvaluation complete! Results saved:")
    print(f"  • JSON: {json_path}")
    print(f"  • CSV:  {csv_path}")
    print(f"  • Report: {report_path}")


if __name__ == "__main__":
    main()
