"""Configurable memory lifecycle policy for Phase 2."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.memory.models import Memory


WORKING = "WORKING"
SHORT_TERM = "SHORT_TERM"
LONG_TERM = "LONG_TERM"
ARCHIVE = "ARCHIVE"


@dataclass(frozen=True)
class LifecyclePolicy:
    """Initial placement and independent lifecycle transition rules."""

    initial_archive_threshold: float = 0.30
    initial_long_term_threshold: float = 0.70
    working_to_long_term_age: timedelta = timedelta(days=7)
    working_to_long_term_min_accesses: int = 1
    short_term_to_archive_age: timedelta = timedelta(days=30)
    archive_compression_level: int = 1

    def initial_tier(self, importance_score: float) -> str:
        score = max(0.0, min(1.0, importance_score))
        if score <= self.initial_archive_threshold:
            return ARCHIVE
        if score >= self.initial_long_term_threshold:
            return WORKING
        return SHORT_TERM

    def transition(self, memory: Memory, now: datetime | None = None) -> str:
        """Return the next tier using lifecycle metadata, not importance."""
        current_time = now or datetime.now(timezone.utc)
        age = current_time - memory.created_at
        if memory.tier == WORKING:
            if (
                age >= self.working_to_long_term_age
                and memory.access_count >= self.working_to_long_term_min_accesses
            ):
                return LONG_TERM
            return WORKING
        if memory.tier == SHORT_TERM:
            if (
                age >= self.short_term_to_archive_age
                or memory.compression_level >= self.archive_compression_level
            ):
                return ARCHIVE
        return memory.tier

    def apply_transition(self, memory: Memory, now: datetime | None = None) -> Memory:
        memory.tier = self.transition(memory, now)
        return memory


TierAssigner = LifecyclePolicy