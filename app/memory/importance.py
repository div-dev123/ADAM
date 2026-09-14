"""Transparent, configurable, and multi-signal importance scoring for ADAM."""

import re
from dataclasses import dataclass
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Filler, Greeting & Boilerplate Detection
# ---------------------------------------------------------------------------

GREETING_TOKENS = {
    "hi", "hello", "hey", "howdy", "heya", "yo", "sup", "greetings",
    "good morning", "good evening", "good afternoon", "good day",
    "what's up", "whats up",
}

ACKNOWLEDGEMENT_TOKENS = {
    "ok", "okay", "k", "kk", "cool", "nice", "awesome", "great", "sure",
    "fine", "yes", "yeah", "yup", "no", "nope", "nah", "thanks", "thank you",
    "thx", "ty", "thanks a lot", "thank you very much", "bye", "goodbye",
    "cya", "see you", "see ya", "test", "testing", "ping", "pong",
    "how are you", "how are you doing", "what are you doing", "sounds good",
    "got it", "understood", "all right", "alright",
}

LLM_BOILERPLATE_PHRASES = (
    "sure!",
    "sure thing!",
    "of course!",
    "that's a great question!",
    "thats a great question!",
    "certainly!",
    "you're welcome!",
    "you are welcome!",
    "no problem!",
    "happy to help!",
    "glad to help!",
    "glad i could help!",
    "how can i help you today?",
    "how can i assist you today?",
    "how can i help you",
    "how can i assist you",
    "let me know if you need anything else",
    "let me know if you have any questions",
    "i can help with that!",
    "i would be happy to help",
    "i am here to help",
    "i'm here to help",
    "as an ai language model",
)


def is_filler(text: str) -> tuple[bool, str]:
    """Determine whether text is a greeting, trivial acknowledgment, or LLM boilerplate.

    Returns (is_filler, reason_tag).
    """
    cleaned = text.strip()
    if not cleaned:
        return True, "empty"

    normalized = re.sub(r"[^\w\s]", "", cleaned.lower()).strip()
    words = normalized.split()
    num_words = len(words)

    if not words:
        return True, "empty"

    # 1. Exact match with known single/multi-word greetings and acknowledgements
    if normalized in GREETING_TOKENS:
        return True, "greeting"
    if normalized in ACKNOWLEDGEMENT_TOKENS:
        return True, "trivial_acknowledgement"

    # 2. Check for LLM boilerplate phrases (substring match without length restriction)
    for phrase in LLM_BOILERPLATE_PHRASES:
        phrase_norm = re.sub(r"[^\w\s]", "", phrase.lower()).strip()
        if normalized == phrase_norm or phrase_norm in normalized:
            return True, "llm_boilerplate"

    # 3. All tokens in a short phrase (<= 4 words) are trivial tokens (e.g., "ok thanks bye")
    all_trivial = GREETING_TOKENS | ACKNOWLEDGEMENT_TOKENS
    if num_words <= 4 and all(w in all_trivial for w in words):
        return True, "trivial_acknowledgement"

    # 4. Standard conversational greeting patterns (e.g. "hi adam", "hey bot", "hello there")
    if re.match(r"^(hi|hello|hey|howdy|greetings)\s+(there|adam|bot|friend|assistant|world)?$", normalized):
        return True, "greeting"

    return False, ""


# ---------------------------------------------------------------------------
# Multi-Signal Importance Scoring
# ---------------------------------------------------------------------------

INTENT_PATTERNS = [
    # Preferences & Emotions
    r"\bi (prefer|like|love|hate|dislike|enjoy|favor|favour|adore)\b",
    r"\b(favorite|favourite|interested in|passionate about)\b",
    # Identity & Background
    r"\b(my name is|i am a|i'm a|i live in|i speak|i work as|my job|my career)\b",
    # Projects, Research & Engineering Work
    r"\b(i am working|i'm working|working on|i am doing|i'm doing|doing a project|project (of|on|about|is|called)|the project is|my project|i am building|i'm building|building a|i develop|developing|i study|i research|my research)\b",
    # Goals, Learning & Directives
    r"\b(my goal is|my goal|i want to|i plan to|i hope to|i need to|i intend to|i am learning|i'm learning|i learn)\b",
    r"\b(remember that|please remember|remember|note that|keep in mind|don't forget|dont forget)\b",
]

# Partial-weight patterns: contribute 0.5 intent (collaborative/team work context)
# These raise score above archive threshold but don't guarantee WORKING tier
PARTIAL_INTENT_PATTERNS = [
    r"\b(we discussed|we decided|we agreed|we planned|team (discussed|decided|agreed|planned|assigned|reviewed))\b",
    r"\b(sprint|standup|retrospective|assigned tasks?|assigned to|teammates?|colleagues?|coworkers?)\b",
    r"\b(meeting|workshop|pair programming|code review|pull request|deployment|release|milestone)\b",
]

SPECIFICITY_KEYWORDS = {
    "dsa", "java", "python", "c++", "rust", "golang", "javascript", "typescript",
    "react", "sql", "mongodb", "sqlite", "postgres", "redis", "docker", "kubernetes",
    "machine learning", "deep learning", "ai", "llm", "neural network", "transformer",
    "memory management", "operating systems", "distributed systems", "system design",
    "algorithm", "algorithms", "data structure", "data structures", "cache", "latency",
    "throughput", "database", "indexing", "concurrency", "async", "api", "rest", "graphql",
    "architecture", "framework", "optimization", "pipeline", "compiler", "kernel",
    "aws", "gcp", "azure", "fastapi", "flask", "django", "git", "linux",
}

EPHEMERAL_PATTERNS = [
    r"\b(today|tonight|yesterday|this morning|this afternoon|this evening)\b",
    r"\b(right now|at the moment|currently raining|for lunch|for breakfast|for dinner)\b",
    r"\b(\d+\s*degrees|weather is|temperature is)\b",
    r"\b(just had|just ate|bought some|heading to)\b",
]

DURABLE_PATTERNS = [
    r"\b(always|never|permanently|fundamental|architecture|core principle|rule|concept|definition)\b",
    r"\b(is defined as|means that|consists of|implements|structured as|focuses on|designed to)\b",
]


@dataclass(frozen=True)
class ImportanceWeights:
    """Configurable weights for multi-signal importance scoring."""

    intent: float = 0.35        # User goals, preferences, identity, explicit directives
    specificity: float = 0.25   # Technical domain terminology, entities, information density
    durability: float = 0.20    # Enduring/architectural facts vs. ephemeral context
    salience: float = 0.10      # Conversational depth & structural complexity
    recurrence: float = 0.05    # Access count frequency
    recency: float = 0.05       # Exponential age decay


class HeuristicImportanceScorer:
    """Multi-signal, interpretable importance scorer combining semantic content signals."""

    def __init__(self, weights: ImportanceWeights | None = None):
        self.weights = weights or ImportanceWeights()

    def score(
        self,
        content: str,
        access_count: int = 0,
        created_at: datetime | None = None,
        now: datetime | None = None,
    ) -> float:
        """Compute an inspectable, bounded [0.0, 1.0] importance score."""
        # 1. Reject greetings and trivial conversational noise
        filler_detected, _ = is_filler(content)
        if filler_detected:
            return 0.0

        text = content.lower().strip()
        words = text.split()
        if not words:
            return 0.0

        weights = self.weights

        # 2. Extract Individual Semantic / Content Signals
        intent_score = self._compute_intent_score(text)
        spec_score = self._compute_specificity_score(content, text, words)
        dur_score = self._compute_durability_score(text, intent_score)
        salience_score = min(len(words) / 15.0, 1.0)
        rec_score = min(max(access_count, 0) / 3.0, 1.0)
        recency_score = self._compute_recency(created_at, now)

        # 3. Calculate Normalized Weighted Sum
        total = (
            weights.intent * intent_score
            + weights.specificity * spec_score
            + weights.durability * dur_score
            + weights.salience * salience_score
            + weights.recurrence * rec_score
            + weights.recency * recency_score
        )

        return max(0.0, min(1.0, round(total, 4)))

    @staticmethod
    def _compute_intent_score(text: str) -> float:
        for pat in INTENT_PATTERNS:
            if re.search(pat, text):
                return 1.0
        # Partial intent for collaborative/team context (raises above archive but not WORKING)
        for pat in PARTIAL_INTENT_PATTERNS:
            if re.search(pat, text):
                return 0.5
        return 0.0

    @staticmethod
    def _compute_specificity_score(raw_text: str, lower_text: str, words: list[str]) -> float:
        kw_matches = sum(1 for kw in SPECIFICITY_KEYWORDS if kw in lower_text)
        acronyms = len(re.findall(r"\b[A-Z]{2,}\b", raw_text))

        if kw_matches >= 2 or (kw_matches >= 1 and acronyms >= 1):
            return 1.0
        if kw_matches == 1 or acronyms >= 1:
            return 0.7
        if len(re.findall(r"\b\d+\b", lower_text)) > 0:
            return 0.4
        if len(words) >= 6 and len(set(words)) / len(words) > 0.75:
            return 0.35
        return 0.1

    @staticmethod
    def _compute_durability_score(text: str, intent_score: float) -> float:
        if any(re.search(pat, text) for pat in EPHEMERAL_PATTERNS):
            return 0.1
        if any(re.search(pat, text) for pat in DURABLE_PATTERNS) or intent_score >= 0.8:
            return 0.85
        return 0.45

    @staticmethod
    def _compute_recency(created_at: datetime | None, now: datetime | None) -> float:
        if created_at is None:
            return 1.0
        current = now or datetime.now(timezone.utc)
        age_seconds = max(0.0, (current - created_at).total_seconds())
        half_life = 86400.0
        return 0.5 ** (age_seconds / half_life)