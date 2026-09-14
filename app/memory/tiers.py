"""Configurable memory tier assignment for Phase 2."""

from dataclasses import dataclass


WORKING = "WORKING"
SHORT_TERM = "SHORT_TERM"
LONG_TERM = "LONG_TERM"
ARCHIVE = "ARCHIVE"


@dataclass(frozen=True)
class TierThresholds:
    archive: float = 0.20
    working: float = 0.45
    short_term: float = 0.70


class TierAssigner:
    """Map a bounded importance score to one of the four memory tiers."""

    def __init__(self, thresholds: TierThresholds | None = None):
        self.thresholds = thresholds or TierThresholds()

    def assign(self, importance_score: float, access_count: int = 0) -> str:
        score = max(0.0, min(1.0, importance_score))
        if score <= self.thresholds.archive:
            return ARCHIVE
        if score <= self.thresholds.working:
            return WORKING
        if score <= self.thresholds.short_term:
            return SHORT_TERM
        return LONG_TERM