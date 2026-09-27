"""Configuration definitions for ADAM baselines and ablation experiments."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ExperimentConfig:
    name: str
    description: str
    is_baseline: bool = False
    ablation_target: Optional[str] = None

    # Core Architectural Component Toggles
    enable_importance_scoring: bool = True
    enable_tier_hierarchy: bool = True
    enable_consolidation: bool = True
    enable_forgetting: bool = True
    enable_query_drift: bool = True
    enable_multi_signal: bool = True
    enable_hybrid_bm25: bool = True
    enable_context_budgeting: bool = True
    enable_redundancy_removal: bool = True

    # Retrieval and budgeting parameters
    top_k: int = 5
    token_budget: int = 500
    redundancy_threshold: float = 0.85


def get_standard_configurations() -> Dict[str, ExperimentConfig]:
    """Returns the standardized suite of 4 baselines and 6 ablation configurations."""
    configs = {}

    # -------------------------------------------------------------------------
    # 1. Configurable Baselines
    # -------------------------------------------------------------------------
    configs["baseline_raw_history"] = ExperimentConfig(
        name="baseline_raw_history",
        description="Raw conversation history baseline: keeps all messages sequentially, no retrieval filtering",
        is_baseline=True,
        enable_importance_scoring=False,
        enable_tier_hierarchy=False,
        enable_consolidation=False,
        enable_forgetting=False,
        enable_query_drift=False,
        enable_multi_signal=False,
        enable_hybrid_bm25=False,
        enable_context_budgeting=False,
        enable_redundancy_removal=False,
    )

    configs["baseline_vector_only"] = ExperimentConfig(
        name="baseline_vector_only",
        description="Dense vector similarity baseline: top-k pure cosine similarity, no metadata signals or drift",
        is_baseline=True,
        enable_importance_scoring=False,
        enable_tier_hierarchy=False,
        enable_consolidation=False,
        enable_forgetting=False,
        enable_query_drift=False,
        enable_multi_signal=False,
        enable_hybrid_bm25=False,
        enable_context_budgeting=False,
        enable_redundancy_removal=False,
    )

    configs["baseline_importance_only"] = ExperimentConfig(
        name="baseline_importance_only",
        description="Importance-based retrieval baseline: ranks memories purely by intrinsic importance score",
        is_baseline=True,
        enable_importance_scoring=True,
        enable_tier_hierarchy=False,
        enable_consolidation=False,
        enable_forgetting=False,
        enable_query_drift=False,
        enable_multi_signal=False,
        enable_hybrid_bm25=False,
        enable_context_budgeting=False,
        enable_redundancy_removal=False,
    )

    configs["adam_full"] = ExperimentConfig(
        name="adam_full",
        description="Full ADAM system: all Phase 1-6 subsystems active (drift, hybrid, 6-signal ranking, budgeting, lifecycle)",
        is_baseline=False,
        enable_importance_scoring=True,
        enable_tier_hierarchy=True,
        enable_consolidation=True,
        enable_forgetting=True,
        enable_query_drift=True,
        enable_multi_signal=True,
        enable_hybrid_bm25=True,
        enable_context_budgeting=True,
        enable_redundancy_removal=True,
    )

    # -------------------------------------------------------------------------
    # 2. Ablations (Single Component Disabled relative to adam_full)
    # -------------------------------------------------------------------------
    configs["ablation_no_importance"] = ExperimentConfig(
        name="ablation_no_importance",
        description="ADAM without importance scoring: uniform 0.5 importance, zero importance ranking weight",
        is_baseline=False,
        ablation_target="importance_scoring",
        enable_importance_scoring=False,
        enable_tier_hierarchy=True,
        enable_consolidation=True,
        enable_forgetting=True,
        enable_query_drift=True,
        enable_multi_signal=True,
        enable_hybrid_bm25=True,
        enable_context_budgeting=True,
        enable_redundancy_removal=True,
    )

    configs["ablation_no_tiers"] = ExperimentConfig(
        name="ablation_no_tiers",
        description="ADAM without tier hierarchy: all memories in single flat tier, zero tier score bonus",
        is_baseline=False,
        ablation_target="tier_hierarchy",
        enable_importance_scoring=True,
        enable_tier_hierarchy=False,
        enable_consolidation=True,
        enable_forgetting=True,
        enable_query_drift=False,  # no tiers implies no tier-scoped drift
        enable_multi_signal=True,
        enable_hybrid_bm25=True,
        enable_context_budgeting=True,
        enable_redundancy_removal=True,
    )

    configs["ablation_no_consolidation"] = ExperimentConfig(
        name="ablation_no_consolidation",
        description="ADAM without consolidation: writes all messages as new items, no duplicate/contradiction handling",
        is_baseline=False,
        ablation_target="consolidation",
        enable_importance_scoring=True,
        enable_tier_hierarchy=True,
        enable_consolidation=False,
        enable_forgetting=True,
        enable_query_drift=True,
        enable_multi_signal=True,
        enable_hybrid_bm25=True,
        enable_context_budgeting=True,
        enable_redundancy_removal=True,
    )

    configs["ablation_no_forgetting"] = ExperimentConfig(
        name="ablation_no_forgetting",
        description="ADAM without forgetting/decay: no Ebbinghaus decay, memories never purged",
        is_baseline=False,
        ablation_target="forgetting",
        enable_importance_scoring=True,
        enable_tier_hierarchy=True,
        enable_consolidation=True,
        enable_forgetting=False,
        enable_query_drift=True,
        enable_multi_signal=True,
        enable_hybrid_bm25=True,
        enable_context_budgeting=True,
        enable_redundancy_removal=True,
    )

    configs["ablation_no_query_drift"] = ExperimentConfig(
        name="ablation_no_query_drift",
        description="ADAM without adaptive query drift: retrieval always searches all tiers without scope narrowing",
        is_baseline=False,
        ablation_target="query_drift",
        enable_importance_scoring=True,
        enable_tier_hierarchy=True,
        enable_consolidation=True,
        enable_forgetting=True,
        enable_query_drift=False,
        enable_multi_signal=True,
        enable_hybrid_bm25=True,
        enable_context_budgeting=True,
        enable_redundancy_removal=True,
    )

    configs["ablation_no_multi_signal"] = ExperimentConfig(
        name="ablation_no_multi_signal",
        description="ADAM without multi-signal ranking: candidates ranked purely by hybrid score without recency, tier, access, or importance weighting",
        is_baseline=False,
        ablation_target="multi_signal_ranking",
        enable_importance_scoring=True,
        enable_tier_hierarchy=True,
        enable_consolidation=True,
        enable_forgetting=True,
        enable_query_drift=True,
        enable_multi_signal=False,
        enable_hybrid_bm25=True,
        enable_context_budgeting=True,
        enable_redundancy_removal=True,
    )

    return configs
