"""Metric computation engine for ADAM evaluation and ablations."""

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class QueryMetricResult:
    query_id: str
    query: str
    latency_ms: float
    retrieved_count: int
    retrieved_tokens: int
    precision: float
    recall: float
    f1: float
    redundant_count: int
    obsolete_facts_leaked: int
    obsolete_facts_total: int
    context_relevance: float
    retrieved_contents: List[str] = field(default_factory=list)


@dataclass
class AggregateExperimentMetrics:
    config_name: str
    description: str
    is_baseline: bool
    ablation_target: Optional[str]

    # Storage and Size Metrics
    total_input_turns: int
    final_stored_memories: int
    storage_reduction_pct: float
    raw_history_total_tokens: int
    avg_context_tokens: float
    context_token_reduction_pct: float

    # Retrieval Quality Metrics
    mean_precision: float
    mean_recall: float
    mean_f1: float
    context_relevance_score: float

    # System Health and Dynamics
    mean_latency_ms: float
    p95_latency_ms: float
    redundant_memory_rate: float
    forgotten_memory_rate: float  # Fraction of obsolete facts successfully excluded (1.0 = perfect suppression)

    # Detailed per-query results
    query_results: List[QueryMetricResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


def estimate_tokens(text: str) -> int:
    """Fast approximation of token count based on whitespace and punctuation."""
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return max(1, int(0.75 * words + 0.25 * (chars / 4.0)))


def _text_matches_fact(text: str, fact: str) -> bool:
    """Case-insensitive fuzzy match for facts and keywords."""
    clean_text = text.lower()
    clean_fact = fact.lower().strip()
    if clean_fact in clean_text:
        return True
    
    # Keyword token matching if fact is multi-word
    fact_words = [w for w in re.findall(r"\b\w+\b", clean_fact) if len(w) > 2]
    if len(fact_words) >= 2:
        overlap = sum(1 for w in fact_words if w in clean_text)
        if overlap / len(fact_words) >= 0.75:
            return True
    return False


def compute_query_metrics(
    query_id: str,
    query_text: str,
    retrieved_contents: List[str],
    target_facts: List[str],
    obsolete_facts: List[str],
    latency_ms: float,
    redundancy_threshold: float = 0.85,
    embedding_func=None,
) -> QueryMetricResult:
    """Computes precision, recall, redundancy, obsolete leakage, and relevance for a single query."""
    retrieved_count = len(retrieved_contents)
    total_tokens = sum(estimate_tokens(c) for c in retrieved_contents)

    # 1. Target Fact Recall & Precision
    if not target_facts:
        # Out-of-scope query: precision is 1.0 if empty, 0.0 if irrelevant memories retrieved
        precision = 1.0 if retrieved_count == 0 else 0.0
        recall = 1.0
        f1 = 1.0 if retrieved_count == 0 else 0.0
    else:
        # Precision: fraction of retrieved memories containing at least one target fact
        relevant_memories = 0
        for content in retrieved_contents:
            if any(_text_matches_fact(content, f) for f in target_facts):
                relevant_memories += 1
        precision = relevant_memories / retrieved_count if retrieved_count > 0 else 0.0

        # Recall: fraction of target facts found in at least one retrieved memory
        matched_facts = 0
        all_retrieved_text = " ".join(retrieved_contents)
        for fact in target_facts:
            if _text_matches_fact(all_retrieved_text, fact):
                matched_facts += 1
        recall = matched_facts / len(target_facts)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # 2. Obsolete Fact Leakage (Testing contradiction/forgetting)
    leaked_obsolete = 0
    all_retrieved_combined = " ".join(retrieved_contents)
    for obs in obsolete_facts:
        if _text_matches_fact(all_retrieved_combined, obs):
            leaked_obsolete += 1

    # 3. Redundancy Detection within Retrieved Set
    redundant_count = 0
    if retrieved_count > 1:
        seen_texts: List[str] = []
        for content in retrieved_contents:
            is_dup = False
            words_a = set(re.findall(r"\b\w+\b", content.lower()))
            for prev in seen_texts:
                words_b = set(re.findall(r"\b\w+\b", prev.lower()))
                union = words_a | words_b
                jaccard = len(words_a & words_b) / len(union) if union else 0.0
                if jaccard >= redundancy_threshold or content.strip() == prev.strip():
                    is_dup = True
                    break
            if is_dup:
                redundant_count += 1
            else:
                seen_texts.append(content)

    # 4. Context Relevance Score
    # Evaluates how densely the retrieved context contains query terms and target facts without obsolete terms
    query_words = set(re.findall(r"\b\w+\b", query_text.lower())) - {"what", "where", "is", "our", "do", "the", "for"}
    if not query_words:
        query_words = {"query"}
    
    retrieved_words = set(re.findall(r"\b\w+\b", all_retrieved_combined.lower()))
    query_coverage = len(query_words & retrieved_words) / len(query_words) if query_words else 0.0
    
    context_relevance = 0.5 * recall + 0.3 * precision + 0.2 * query_coverage
    if leaked_obsolete > 0 and obsolete_facts:
        context_relevance *= max(0.0, 1.0 - (leaked_obsolete / len(obsolete_facts)))

    return QueryMetricResult(
        query_id=query_id,
        query=query_text,
        latency_ms=round(latency_ms, 3),
        retrieved_count=retrieved_count,
        retrieved_tokens=total_tokens,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        redundant_count=redundant_count,
        obsolete_facts_leaked=leaked_obsolete,
        obsolete_facts_total=len(obsolete_facts),
        context_relevance=round(context_relevance, 4),
        retrieved_contents=retrieved_contents,
    )


def aggregate_metrics(
    config_name: str,
    description: str,
    is_baseline: bool,
    ablation_target: Optional[str],
    total_input_turns: int,
    final_stored_memories: int,
    raw_history_total_tokens: int,
    query_results: List[QueryMetricResult],
) -> AggregateExperimentMetrics:
    """Aggregates per-query metric measurements into a comprehensive experiment summary."""
    storage_red = (
        (total_input_turns - final_stored_memories) / total_input_turns * 100.0
        if total_input_turns > 0
        else 0.0
    )

    avg_tokens = (
        float(np.mean([q.retrieved_tokens for q in query_results]))
        if query_results
        else 0.0
    )

    token_red = (
        (raw_history_total_tokens - avg_tokens) / raw_history_total_tokens * 100.0
        if raw_history_total_tokens > 0
        else 0.0
    )

    precisions = [q.precision for q in query_results]
    recalls = [q.recall for q in query_results]
    f1s = [q.f1 for q in query_results]
    relevances = [q.context_relevance for q in query_results]
    latencies = [q.latency_ms for q in query_results]

    total_retrieved = sum(q.retrieved_count for q in query_results)
    total_redundant = sum(q.redundant_count for q in query_results)
    redundancy_rate = (
        (total_redundant / total_retrieved * 100.0) if total_retrieved > 0 else 0.0
    )

    total_obsolete_possible = sum(q.obsolete_facts_total for q in query_results)
    total_obsolete_leaked = sum(q.obsolete_facts_leaked for q in query_results)
    if total_obsolete_possible > 0:
        forgotten_rate = (
            (total_obsolete_possible - total_obsolete_leaked)
            / total_obsolete_possible
            * 100.0
        )
    else:
        forgotten_rate = 100.0

    return AggregateExperimentMetrics(
        config_name=config_name,
        description=description,
        is_baseline=is_baseline,
        ablation_target=ablation_target,
        total_input_turns=total_input_turns,
        final_stored_memories=final_stored_memories,
        storage_reduction_pct=round(storage_red, 2),
        raw_history_total_tokens=raw_history_total_tokens,
        avg_context_tokens=round(avg_tokens, 1),
        context_token_reduction_pct=round(token_red, 2),
        mean_precision=round(float(np.mean(precisions)), 4) if precisions else 0.0,
        mean_recall=round(float(np.mean(recalls)), 4) if recalls else 0.0,
        mean_f1=round(float(np.mean(f1s)), 4) if f1s else 0.0,
        context_relevance_score=round(float(np.mean(relevances)), 4) if relevances else 0.0,
        mean_latency_ms=round(float(np.mean(latencies)), 2) if latencies else 0.0,
        p95_latency_ms=round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        redundant_memory_rate=round(redundancy_rate, 2),
        forgotten_memory_rate=round(forgotten_rate, 2),
        query_results=query_results,
    )
