"""Transparent, replaceable importance scoring for Phase 2."""

from dataclasses import dataclass
from datetime import datetime, timezone


PERSISTENT_MARKERS = (
    "my name is",
    "i prefer",
    "i like",
    "i love",
    "i hate",
    "i work",
    "my goal is",
    "i am learning",
    "i use",
    "remember that",
    "please remember",
)


@dataclass(frozen=True)
class ImportanceWeights:
    persistent: float = 0.55
    length: float = 0.20
    recurrence: float = 0.15
    recency: float = 0.10


class HeuristicImportanceScorer:
    """Score memories with inspectable text and metadata signals.

    This is deliberately not an LLM judge. Each component is bounded in
    [0, 1], making the final weighted score reproducible and easy to ablate.
    """

    def __init__(self, weights: ImportanceWeights | None = None):
        self.weights = weights or ImportanceWeights()

    def score(
        self,
        content: str,
        access_count: int = 0,
        created_at: datetime | None = None,
        now: datetime | None = None,
    ) -> float:
        text = content.lower().strip()
        persistent = float(any(marker in text for marker in PERSISTENT_MARKERS))
        length = min(len(text.split()) / 20.0, 1.0)
        recurrence = min(max(access_count, 0) / 3.0, 1.0)
        recency = self._recency(created_at, now)
        weights = self.weights
        total = (
            weights.persistent * persistent
            + weights.length * length
            + weights.recurrence * recurrence
            + weights.recency * recency
        )
        return max(0.0, min(1.0, total))

    @staticmethod
    def _recency(created_at: datetime | None, now: datetime | None) -> float:
        if created_at is None:
            return 1.0
        current = now or datetime.now(timezone.utc)
        age_seconds = max(0.0, (current - created_at).total_seconds())
        half_life = 86400.0
        return 0.5 ** (age_seconds / half_life)