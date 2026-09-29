import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.memory.models import Memory, utc_now
from app.memory.storage import SQLiteStorage
from app.memory.tiers import ARCHIVE, LONG_TERM, SHORT_TERM, WORKING


@dataclass
class LifecyclePolicyConfig:
    """Configurable parameters governing memory retention, transitions, and selective forgetting."""

    forgetting_threshold: float = 0.75
    protected_importance: float = 0.70
    protected_access_count: int = 3
    frequent_access_boost_threshold: int = 3

    # Age thresholds
    working_to_long_term_age: timedelta = timedelta(days=7)
    working_to_long_term_importance_min: float = 0.60
    working_inactivity_threshold: timedelta = timedelta(days=5)

    short_term_to_archive_age: timedelta = timedelta(days=30)
    archive_obsolete_age: timedelta = timedelta(days=60)

    # Ebbinghaus recency decay rate lambda (half-life = ln(2)/lambda ~ 14 days at lambda=0.05)
    recency_decay_lambda: float = 0.05

    # Weights for Retention Score R in [0, 1]
    weight_importance: float = 0.40
    weight_recency: float = 0.30
    weight_frequency: float = 0.30
    compression_penalty_per_level: float = 0.05


@dataclass
class ForgettingScoreBreakdown:
    forgetting_score: float
    retention_score: float
    importance_component: float
    recency_component: float
    frequency_component: float
    compression_penalty: float
    age_days: float
    inactive_days: float


@dataclass
class LifecycleDecision:
    action: str  # "NONE", "TRANSITION", "FORGET", "PROTECT"
    memory_id: str
    from_tier: str
    to_tier: Optional[str] = None
    reason: str = ""
    forgetting_score: float = 0.0
    breakdown: Optional[ForgettingScoreBreakdown] = None
    is_protected: bool = False


class MemoryLifecycleManager:
    """Evaluates memory state, applies tier transitions, and selectively forgets obsolete memories."""

    def __init__(
        self,
        storage: SQLiteStorage,
        config: Optional[LifecyclePolicyConfig] = None,
        compression_service=None,
    ):
        self.storage = storage
        self.config = config or LifecyclePolicyConfig()
        self.compression_service = compression_service

    def calculate_forgetting_score(
        self, memory: Memory, now: Optional[datetime] = None
    ) -> Tuple[float, ForgettingScoreBreakdown]:
        """Compute the mathematical retention and forgetting scores for a memory."""
        current_time = now or utc_now()

        age_seconds = max(0.0, (current_time - memory.created_at).total_seconds())
        age_days = age_seconds / 86400.0

        inactive_seconds = max(
            0.0, (current_time - memory.last_accessed).total_seconds()
        )
        inactive_days = inactive_seconds / 86400.0

        # Normalized signals
        norm_importance = max(0.0, min(1.0, memory.importance_score))

        # Ebbinghaus exponential recency decay
        recency_factor = math.exp(-self.config.recency_decay_lambda * inactive_days)

        # Logarithmic access frequency boost (0 accesses -> 0.0, 3 accesses -> 0.58, 10 accesses -> 1.0)
        target_accesses = max(5, self.config.protected_access_count * 2)
        frequency_factor = min(
            1.0, math.log(1.0 + memory.access_count) / math.log(1.0 + target_accesses)
        )

        compression_penalty = (
            memory.compression_level * self.config.compression_penalty_per_level
        )

        importance_comp = self.config.weight_importance * norm_importance
        recency_comp = self.config.weight_recency * recency_factor
        frequency_comp = self.config.weight_frequency * frequency_factor

        retention_score = max(
            0.0,
            min(
                1.0,
                importance_comp + recency_comp + frequency_comp - compression_penalty,
            ),
        )
        forgetting_score = round(max(0.0, min(1.0, 1.0 - retention_score)), 4)

        breakdown = ForgettingScoreBreakdown(
            forgetting_score=forgetting_score,
            retention_score=round(retention_score, 4),
            importance_component=round(importance_comp, 4),
            recency_component=round(recency_comp, 4),
            frequency_component=round(frequency_comp, 4),
            compression_penalty=round(compression_penalty, 4),
            age_days=round(age_days, 2),
            inactive_days=round(inactive_days, 2),
        )
        return forgetting_score, breakdown

    def evaluate_memory(
        self, memory: Memory, now: Optional[datetime] = None
    ) -> LifecycleDecision:
        """Evaluate a memory against retention, transition, and forgetting policies."""
        current_time = now or utc_now()
        forgetting_score, breakdown = self.calculate_forgetting_score(memory, current_time)

        # 1. Anti-amnesia protection check
        # High-importance memories or frequently accessed memories are guarded against deletion
        is_high_importance = memory.importance_score >= self.config.protected_importance
        is_frequently_accessed = (
            memory.access_count >= self.config.protected_access_count
        )
        is_superseded = bool(memory.superseded_by)

        is_protected = (is_high_importance or is_frequently_accessed) and not is_superseded

        # 2. Check for Selective Forgetting (Deletion)
        if not is_protected:
            # Memory must be in ARCHIVE, or superseded, with high forgetting score
            can_forget = False
            forget_reason = ""

            if is_superseded:
                # Superseded memories with high forgetting score
                if forgetting_score >= self.config.forgetting_threshold:
                    can_forget = True
                    forget_reason = (
                        f"Superseded by memory {memory.superseded_by} and high forgetting score "
                        f"({forgetting_score:.2f} >= {self.config.forgetting_threshold})"
                    )
            elif memory.tier == ARCHIVE:
                # Archived memory that is old or low value with high forgetting score
                archive_age_threshold = self.config.archive_obsolete_age.total_seconds() / 86400.0
                if (
                    forgetting_score >= self.config.forgetting_threshold
                    and (
                        breakdown.age_days >= archive_age_threshold
                        or memory.importance_score <= 0.25
                        or breakdown.inactive_days >= archive_age_threshold
                    )
                ):
                    can_forget = True
                    forget_reason = (
                        f"Obsolete archived memory with low retention "
                        f"(forgetting score {forgetting_score:.2f} >= {self.config.forgetting_threshold}, "
                        f"importance {memory.importance_score:.2f}, inactive {breakdown.inactive_days:.1f}d)"
                    )

            if can_forget:
                return LifecycleDecision(
                    action="FORGET",
                    memory_id=memory.memory_id,
                    from_tier=memory.tier,
                    to_tier=None,
                    reason=forget_reason,
                    forgetting_score=forgetting_score,
                    breakdown=breakdown,
                    is_protected=False,
                )

        # 3. Check for Tier Transitions
        age = current_time - memory.created_at
        inactive = current_time - memory.last_accessed

        # Transition: WORKING -> LONG_TERM
        # Working memory moves to LONG_TERM when it becomes older/inactive but remains important
        if memory.tier == WORKING:
            # Check if frequently accessed recently - if so, it stays in WORKING longer!
            has_frequent_activity = (
                memory.access_count >= self.config.frequent_access_boost_threshold
                and inactive < self.config.working_inactivity_threshold
            )
            if not has_frequent_activity:
                if (
                    age >= self.config.working_to_long_term_age
                    and memory.importance_score >= self.config.working_to_long_term_importance_min
                    and memory.access_count >= 1
                ):
                    return LifecycleDecision(
                        action="TRANSITION",
                        memory_id=memory.memory_id,
                        from_tier=WORKING,
                        to_tier=LONG_TERM,
                        reason=(
                            f"Aged working memory with sustained importance "
                            f"(age {breakdown.age_days:.1f}d >= {self.config.working_to_long_term_age.days}d, "
                            f"importance {memory.importance_score:.2f} >= {self.config.working_to_long_term_importance_min})"
                        ),
                        forgetting_score=forgetting_score,
                        breakdown=breakdown,
                        is_protected=is_protected,
                    )
                elif age >= self.config.working_to_long_term_age and memory.importance_score < self.config.working_to_long_term_importance_min:
                    # Low-importance working memory that aged and wasn't accessed moves to SHORT_TERM
                    return LifecycleDecision(
                        action="TRANSITION",
                        memory_id=memory.memory_id,
                        from_tier=WORKING,
                        to_tier=SHORT_TERM,
                        reason=(
                            f"Aging working memory with low importance demoted to SHORT_TERM "
                            f"(importance {memory.importance_score:.2f} < {self.config.working_to_long_term_importance_min})"
                        ),
                        forgetting_score=forgetting_score,
                        breakdown=breakdown,
                        is_protected=is_protected,
                    )

        # Transition: SHORT_TERM -> ARCHIVE
        # SHORT_TERM moves to ARCHIVE as it becomes older and/or less useful
        if memory.tier == SHORT_TERM:
            # Frequently accessed memories stay in SHORT_TERM longer
            is_frequently_active = (
                memory.access_count >= self.config.frequent_access_boost_threshold
                and inactive < self.config.short_term_to_archive_age
            )
            if not is_frequently_active:
                if (
                    age >= self.config.short_term_to_archive_age
                    or breakdown.retention_score < 0.40
                    or memory.compression_level >= 1
                ):
                    return LifecycleDecision(
                        action="TRANSITION",
                        memory_id=memory.memory_id,
                        from_tier=SHORT_TERM,
                        to_tier=ARCHIVE,
                        reason=(
                            f"Short-term memory demoted to ARCHIVE "
                            f"(age {breakdown.age_days:.1f}d, retention score {breakdown.retention_score:.2f})"
                        ),
                        forgetting_score=forgetting_score,
                        breakdown=breakdown,
                        is_protected=is_protected,
                    )

        # Protection or retention notice
        if is_protected:
            prot_reasons = []
            if is_high_importance:
                prot_reasons.append(f"high importance ({memory.importance_score:.2f} >= {self.config.protected_importance})")
            if is_frequently_accessed:
                prot_reasons.append(f"frequently accessed ({memory.access_count} >= {self.config.protected_access_count})")
            return LifecycleDecision(
                action="PROTECT",
                memory_id=memory.memory_id,
                from_tier=memory.tier,
                to_tier=memory.tier,
                reason="Protected: " + ", ".join(prot_reasons),
                forgetting_score=forgetting_score,
                breakdown=breakdown,
                is_protected=True,
            )

        return LifecycleDecision(
            action="NONE",
            memory_id=memory.memory_id,
            from_tier=memory.tier,
            to_tier=memory.tier,
            reason="Active memory within acceptable lifecycle thresholds",
            forgetting_score=forgetting_score,
            breakdown=breakdown,
            is_protected=False,
        )

    def run_lifecycle_pass(
        self,
        user_id: Optional[str] = None,
        dry_run: bool = False,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Execute a full lifecycle evaluation pass over all memories."""
        current_time = now or utc_now()
        memories = self.storage.get_all_memories(user_id=user_id)

        transitions = []
        forgotten = []
        protected = []
        unchanged = []

        for memory in memories:
            decision = self.evaluate_memory(memory, current_time)

            if decision.action == "FORGET":
                forgotten_item = {
                    "memory_id": memory.memory_id,
                    "user_id": memory.user_id,
                    "content": memory.content,
                    "tier": memory.tier,
                    "importance_score": memory.importance_score,
                    "access_count": memory.access_count,
                    "forgetting_score": decision.forgetting_score,
                    "reason": decision.reason,
                }
                if not dry_run:
                    self.storage.forget_memory(memory.memory_id, reason=decision.reason)
                forgotten.append(forgotten_item)

            elif decision.action == "TRANSITION":
                transition_item = {
                    "memory_id": memory.memory_id,
                    "user_id": memory.user_id,
                    "from_tier": decision.from_tier,
                    "to_tier": decision.to_tier,
                    "importance_score": memory.importance_score,
                    "access_count": memory.access_count,
                    "reason": decision.reason,
                }
                if not dry_run and decision.to_tier:
                    old_tier = memory.tier
                    if self.compression_service:
                        self.compression_service.transition(memory, decision.to_tier)
                    else:
                        memory.tier = decision.to_tier
                        memory.updated_at = current_time
                        self.storage.update_memory(memory)
                        self.storage.record_history(
                            memory=memory,
                            operation="TIER_TRANSITION",
                            old_content=f"tier:{old_tier}",
                            new_content=f"tier:{decision.to_tier}",
                            reason=decision.reason,
                        )
                transitions.append(transition_item)

            elif decision.action == "PROTECT":
                protected.append({
                    "memory_id": memory.memory_id,
                    "user_id": memory.user_id,
                    "tier": memory.tier,
                    "importance_score": memory.importance_score,
                    "access_count": memory.access_count,
                    "reason": decision.reason,
                })

            else:
                unchanged.append({
                    "memory_id": memory.memory_id,
                    "user_id": memory.user_id,
                    "tier": memory.tier,
                    "forgetting_score": decision.forgetting_score,
                })

        return {
            "status": "completed",
            "dry_run": dry_run,
            "evaluated_count": len(memories),
            "transition_count": len(transitions),
            "forgotten_count": len(forgotten),
            "protected_count": len(protected),
            "unchanged_count": len(unchanged),
            "transitions": transitions,
            "forgotten": forgotten,
            "protected": protected,
            "timestamp": current_time.isoformat(),
        }
