"""Transparent, replaceable importance scoring for Phase 2."""

from dataclasses import dataclass
from datetime import datetime, timezone


PERSISTENT_MARKERS = (
    # Personal Identity & State
    "my name is",
    "i am a",
    "i'm a",
    "i live",
    "i speak",
    "i have",
    
    # Preferences & Emotions
    "i prefer",
    "i like",
    "i love",
    "i hate",
    "i dislike",
    "i enjoy",
    "favorite",
    "favourite",
    "interested in",
    "passionate about",

    # Projects, Work & Activities
    "i work",
    "i am working",
    "i'm working",
    "working on",
    "i am doing",
    "i'm doing",
    "doing a project",
    "project of",
    "project on",
    "project about",
    "my project",
    "i am building",
    "i'm building",
    "building a",
    "developing",
    "i develop",
    "i study",
    "i am studying",
    "i research",
    "my research",
    "my job",
    "my career",

    # Skills, Languages & Technical Domains
    "dsa",
    "data structures",
    "algorithms",
    "java",
    "python",
    "c++",
    "rust",
    "golang",
    "javascript",
    "typescript",
    "react",
    "sql",
    "mongodb",
    "sqlite",
    "machine learning",
    "deep learning",
    "ai",
    "memory management",
    "operating systems",
    "distributed systems",
    "system design",

    # Goals, Intent & Learning
    "my goal is",
    "my goal",
    "i want to",
    "i plan to",
    "i hope to",
    "i am learning",
    "i'm learning",
    "i learn",
    "i use",
    "i code",
    "i program",
    "i know",

    # Explicit Memory Directives
    "remember that",
    "please remember",
    "remember",
    "note that",
    "keep in mind",
    "don't forget",
)

TRIVIAL_NOISE_TOKENS = {
    "hi", "hello", "hey", "ok", "okay", "k", "cool", "sure",
    "yes", "yeah", "yup", "no", "nope", "thanks", "thank you",
    "bye", "goodbye", "good morning", "good evening", "test"
}


@dataclass(frozen=True)
class ImportanceWeights:
    persistent: float = 0.50
    base_salience: float = 0.25
    length: float = 0.15
    recurrence: float = 0.10
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
        words = text.split()
        num_words = len(words)

        if not text:
            return 0.0

        # Detect trivial low-value noise/greetings
        if num_words <= 3 and (text in TRIVIAL_NOISE_TOKENS or all(w in TRIVIAL_NOISE_TOKENS for w in words)):
            return 0.10

        persistent = float(any(marker in text for marker in PERSISTENT_MARKERS))
        length = min(num_words / 20.0, 1.0)
        recurrence = min(max(access_count, 0) / 3.0, 1.0)
        recency = self._recency(created_at, now)
        weights = self.weights

        base_val = getattr(weights, "base_salience", 0.25)
        total = (
            weights.persistent * persistent
            + base_val
            + weights.length * length
            + weights.recurrence * recurrence
            + weights.recency * recency
        )
        return max(0.0, min(1.0, round(total, 4)))

    @staticmethod
    def _recency(created_at: datetime | None, now: datetime | None) -> float:
        if created_at is None:
            return 1.0
        current = now or datetime.now(timezone.utc)
        age_seconds = max(0.0, (current - created_at).total_seconds())
        half_life = 86400.0
        return 0.5 ** (age_seconds / half_life)