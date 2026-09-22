import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from app.memory.tiers import ARCHIVE, LONG_TERM, SHORT_TERM, WORKING
from app.retrieval.similarity import cosine_similarity


@dataclass
class DriftConfig:
    """Configurable thresholds and tier scope mappings for query drift detection."""

    low_drift_threshold: float = 0.55
    high_drift_threshold: float = 0.85
    time_weight: float = 0.20
    time_half_life_hours: float = 1.0

    low_drift_tiers: Tuple[str, ...] = (WORKING, SHORT_TERM)
    medium_drift_tiers: Tuple[str, ...] = (WORKING, SHORT_TERM, LONG_TERM)
    high_drift_tiers: Tuple[str, ...] = (WORKING, SHORT_TERM, LONG_TERM, ARCHIVE)
    default_scope_tiers: Tuple[str, ...] = (WORKING, SHORT_TERM)


@dataclass
class DriftResult:
    """Comprehensive telemetry describing detected query drift and adaptive scope selection."""

    drift_score: float
    drift_level: str  # "LOW", "MEDIUM", "HIGH"
    semantic_distance: float
    time_factor: float
    scope_tiers: List[str]
    reason: str
    context_text: Optional[str] = None


class QueryDriftDetector:
    """Detects semantic and temporal topic drift between user query and conversation context."""

    def __init__(self, embeddings, config: Optional[DriftConfig] = None):
        self.embeddings = embeddings
        self.config = config or DriftConfig()

    def extract_context_text(
        self, context: Optional[Union[str, List[Dict[str, Any]]]]
    ) -> Optional[str]:
        """Extract a coherent textual representation from a raw string or chat history list."""
        if not context:
            return None
        if isinstance(context, str):
            cleaned = context.strip()
            return cleaned if cleaned else None
        if isinstance(context, list):
            # Extract content from recent turns (up to 3 recent messages)
            recent_turns = []
            for item in context[-3:]:
                if isinstance(item, dict) and "content" in item and item["content"]:
                    content = str(item["content"]).strip()
                    if content:
                        role = item.get("role", "user")
                        recent_turns.append(f"{role}: {content}")
                elif isinstance(item, str) and item.strip():
                    recent_turns.append(item.strip())
            if recent_turns:
                return " \n ".join(recent_turns)
        return None

    def detect_drift(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        context: Optional[Union[str, List[Dict[str, Any]]]] = None,
        context_time: Optional[datetime] = None,
        current_time: Optional[datetime] = None,
    ) -> DriftResult:
        """Calculate query drift and determine the optimal memory scope."""
        context_text = self.extract_context_text(context)

        # 1. No context case: Initial turn or unconditioned search
        if not context_text:
            return DriftResult(
                drift_score=0.0,
                drift_level="LOW",
                semantic_distance=0.0,
                time_factor=0.0,
                scope_tiers=list(self.config.default_scope_tiers),
                reason="No previous conversation context; default focused scope selected (WORKING, SHORT_TERM).",
                context_text=None,
            )

        # 2. Semantic distance computation
        if query_embedding is None:
            query_embedding = self.embeddings.encode(query)
        context_embedding = self.embeddings.encode(context_text)

        similarity = float(cosine_similarity(query_embedding, context_embedding))
        semantic_distance = max(0.0, min(1.0, 1.0 - similarity))

        # 3. Temporal staleness computation (if context_time provided)
        time_factor = 0.0
        if context_time is not None:
            now = current_time or datetime.now(timezone.utc)
            delta_seconds = max(0.0, (now - context_time).total_seconds())
            delta_hours = delta_seconds / 3600.0
            half_life = max(0.01, self.config.time_half_life_hours)
            time_factor = 1.0 - math.exp(-0.6931 * delta_hours / half_life)
            raw_score = (
                (1.0 - self.config.time_weight) * semantic_distance
                + self.config.time_weight * time_factor
            )
        else:
            raw_score = semantic_distance

        drift_score = round(max(0.0, min(1.0, raw_score)), 4)
        sem_dist_rounded = round(semantic_distance, 4)
        time_factor_rounded = round(time_factor, 4)

        # 4. Scope determination based on configured thresholds
        if drift_score < self.config.low_drift_threshold:
            drift_level = "LOW"
            scope_tiers = list(self.config.low_drift_tiers)
            reason = (
                f"Low query drift ({drift_score:.2f} < {self.config.low_drift_threshold:.2f}); "
                f"topic strongly aligns with recent context. Prioritizing {', '.join(scope_tiers)} tiers."
            )
        elif drift_score < self.config.high_drift_threshold:
            drift_level = "MEDIUM"
            scope_tiers = list(self.config.medium_drift_tiers)
            reason = (
                f"Medium query drift ({drift_score:.2f}); topic evolution detected. "
                f"Expanding scope to include {', '.join(scope_tiers)}."
            )
        else:
            drift_level = "HIGH"
            scope_tiers = list(self.config.high_drift_tiers)
            reason = (
                f"High query drift ({drift_score:.2f} >= {self.config.high_drift_threshold:.2f}); "
                f"significant topic shift detected. Broadening scope across all tiers: {', '.join(scope_tiers)}."
            )

        return DriftResult(
            drift_score=drift_score,
            drift_level=drift_level,
            semantic_distance=sem_dist_rounded,
            time_factor=time_factor_rounded,
            scope_tiers=scope_tiers,
            reason=reason,
            context_text=context_text,
        )
