"""Multi-signal ranking engine for ADAM Phase 6: Multi-Signal Retrieval.

Combines semantic vector similarity, lexical query relevance (BM25), intrinsic importance,
recency decay, access frequency, and operational memory tier into an explainable,
configurable ranking score for memory candidates.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.memory.models import Memory, utc_now
from app.memory.tiers import ARCHIVE, LONG_TERM, SHORT_TERM, WORKING
from app.retrieval.similarity import cosine_similarity


DEFAULT_TIER_SCORES: Dict[str, float] = {
    WORKING: 1.0,
    LONG_TERM: 0.85,
    SHORT_TERM: 0.70,
    ARCHIVE: 0.40,
}


@dataclass
class RankingWeights:
    """Configurable weights and tuning parameters for multi-signal retrieval ranking.

    Allows ablation studies by zeroing any weight (e.g. semantic_similarity=1.0, others=0.0).
    """

    semantic_similarity: float = 0.35
    query_relevance: float = 0.20
    importance: float = 0.15
    recency: float = 0.10
    access_frequency: float = 0.10
    tier: float = 0.10

    # Tuning parameters
    tier_scores: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TIER_SCORES))
    recency_half_life_days: float = 7.0
    access_saturation_count: int = 5
    bm25_saturation_scale: float = 2.0


class MultiSignalRanker:
    """Evaluates candidate memories across 6 distinct signals and computes a unified ranking score."""

    def __init__(self, weights: Optional[RankingWeights] = None):
        self.weights = weights or RankingWeights()

    def calculate_recency_score(
        self,
        memory: Memory,
        now: Optional[datetime] = None,
    ) -> float:
        """Calculate exponential recency decay [0.0, 1.0] using last_accessed or created_at."""
        current_time = now or utc_now()
        ref_time = memory.last_accessed or memory.created_at
        age_seconds = max(0.0, (current_time - ref_time).total_seconds())
        age_days = age_seconds / 86400.0
        half_life = max(0.01, self.weights.recency_half_life_days)
        decay = math.exp(-0.6931 * (age_days / half_life))
        return max(0.0, min(1.0, decay))

    def calculate_frequency_score(self, memory: Memory) -> float:
        """Calculate normalized access frequency [0.0, 1.0] with logarithmic saturation."""
        target = max(1, self.weights.access_saturation_count)
        count = max(0, memory.access_count)
        score = math.log(1.0 + count) / math.log(1.0 + target)
        return max(0.0, min(1.0, score))

    def calculate_tier_score(self, memory: Memory) -> float:
        """Calculate normalized tier score [0.0, 1.0]."""
        tier = memory.tier or WORKING
        return max(0.0, min(1.0, self.weights.tier_scores.get(tier, 0.5)))

    def calculate_query_relevance_score(self, bm25_score: float) -> float:
        """Normalize BM25 lexical score to [0.0, 1.0] using a soft-saturation curve."""
        if bm25_score <= 0.0:
            return 0.0
        scale = max(0.1, self.weights.bm25_saturation_scale)
        norm = bm25_score / (bm25_score + scale)
        return max(0.0, min(1.0, norm))

    def score_candidate(
        self,
        memory: Memory,
        dense_score: float,
        bm25_score: float,
        rrf_score: float = 0.0,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Compute all 6 individual normalized signals and the combined ranking score."""
        w = self.weights

        # 1. Individual signal normalization [0.0, 1.0]
        s_sem = max(0.0, min(1.0, float(dense_score)))
        s_rel = self.calculate_query_relevance_score(float(bm25_score))
        s_imp = max(0.0, min(1.0, float(memory.importance_score)))
        s_rec = self.calculate_recency_score(memory, now=now)
        s_freq = self.calculate_frequency_score(memory)
        s_tier = self.calculate_tier_score(memory)

        # 2. Weighted contributions
        signals = {
            "semantic_similarity": {
                "value": round(s_sem, 4),
                "weight": w.semantic_similarity,
                "contribution": round(w.semantic_similarity * s_sem, 4),
                "raw_value": round(float(dense_score), 4),
            },
            "query_relevance": {
                "value": round(s_rel, 4),
                "weight": w.query_relevance,
                "contribution": round(w.query_relevance * s_rel, 4),
                "raw_value": round(float(bm25_score), 4),
            },
            "importance": {
                "value": round(s_imp, 4),
                "weight": w.importance,
                "contribution": round(w.importance * s_imp, 4),
                "raw_value": round(float(memory.importance_score), 4),
            },
            "recency": {
                "value": round(s_rec, 4),
                "weight": w.recency,
                "contribution": round(w.recency * s_rec, 4),
                "raw_value": (
                    (now or utc_now()) - (memory.last_accessed or memory.created_at)
                ).total_seconds()
                / 86400.0,
            },
            "access_frequency": {
                "value": round(s_freq, 4),
                "weight": w.access_frequency,
                "contribution": round(w.access_frequency * s_freq, 4),
                "raw_value": memory.access_count,
            },
            "tier": {
                "value": round(s_tier, 4),
                "weight": w.tier,
                "contribution": round(w.tier * s_tier, 4),
                "raw_value": memory.tier,
            },
        }

        total_weight = sum(s["weight"] for s in signals.values())
        if total_weight > 0.0:
            weighted_sum = sum(s["contribution"] for s in signals.values())
            final_score = round(max(0.0, min(1.0, weighted_sum / total_weight)), 4)
        else:
            final_score = round(s_sem, 4)

        # 3. Generate human-readable selection reason
        selection_reason = self._generate_selection_reason(signals, memory, final_score)

        return {
            "memory": memory,
            "final_score": final_score,
            "ranking_score": final_score,
            "similarity": final_score,  # Backwards compatibility for UI and adapters
            "dense_score": float(dense_score),
            "bm25_score": float(bm25_score),
            "rrf_score": float(rrf_score),
            "signals": signals,
            "signal_scores": {k: v["value"] for k, v in signals.items()},
            "selection_reason": selection_reason,
            "memory_tier": memory.tier,
            "importance": memory.importance_score,
        }

    def _generate_selection_reason(
        self, signals: Dict[str, Dict[str, Any]], memory: Memory, final_score: float
    ) -> str:
        """Create an intuitive rationale explaining why this memory ranked high."""
        # Find the two highest positive contributions
        sorted_contribs = sorted(
            signals.items(),
            key=lambda x: x[1]["contribution"],
            reverse=True,
        )
        top_reasons = []
        for name, sig in sorted_contribs[:2]:
            if sig["contribution"] > 0.0:
                if name == "semantic_similarity":
                    top_reasons.append(f"high semantic match ({sig['value']:.2f})")
                elif name == "query_relevance":
                    top_reasons.append(f"lexical keyword match ({sig['value']:.2f})")
                elif name == "importance":
                    top_reasons.append(f"high importance ({memory.importance_score:.2f})")
                elif name == "recency":
                    top_reasons.append("recent activity")
                elif name == "access_frequency":
                    top_reasons.append(f"frequent access ({memory.access_count}x)")
                elif name == "tier":
                    top_reasons.append(f"active {memory.tier} tier")

        reason_text = " and ".join(top_reasons) if top_reasons else "baseline relevance"
        return f"Ranked #{final_score:.2f} based on {reason_text}."

    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        now: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Score and sort candidate memories by final multi-signal score descending."""
        scored = []
        for cand in candidates:
            memory = cand["memory"]
            dense = cand.get("dense_score", cand.get("similarity", 0.0))
            bm25 = cand.get("bm25_score", 0.0)
            rrf = cand.get("rrf_score", 0.0)

            scored_item = self.score_candidate(
                memory=memory,
                dense_score=dense,
                bm25_score=bm25,
                rrf_score=rrf,
                now=now,
            )
            # Preserve existing diagnostic keys if present
            if "dense_rank" in cand:
                scored_item["dense_rank"] = cand["dense_rank"]
            if "bm25_rank" in cand:
                scored_item["bm25_rank"] = cand["bm25_rank"]

            scored.append(scored_item)

        # Sort descending by final score, tiebreak with dense_score and importance
        scored.sort(
            key=lambda x: (
                x["final_score"],
                x["dense_score"],
                x["memory"].importance_score,
            ),
            reverse=True,
        )
        return scored
