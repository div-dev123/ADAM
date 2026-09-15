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

    # 2. Check for LLM boilerplate phrases
    #    - Short phrases (1-3 words, e.g. "certainly", "sure"): exact match only.
    #      They appear naturally in long informative responses so substring is too aggressive.
    #    - Long phrases (4+ words, e.g. "how can i help you today"): substring match.
    #      These are specific enough that their presence alone marks boilerplate.
    for phrase in LLM_BOILERPLATE_PHRASES:
        phrase_norm = re.sub(r"[^\w\s]", "", phrase.lower()).strip()
        phrase_words = phrase_norm.split()
        if len(phrase_words) <= 3:
            # Only flag if the ENTIRE message is this phrase
            if normalized == phrase_norm:
                return True, "llm_boilerplate"
        else:
            # Long specific phrase: flag if it appears anywhere in the message
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

# Graduated intent categories with specific weights
INTENT_CATEGORY_PATTERNS = [
    # Explicit user directives to the system (strength: 1.0)
    (
        "directive",
        1.0,
        [
            r"\b(remember that|please remember|remember this|note that|keep in mind|don't forget|dont forget)\b",
            r"\b(store this|save this|record this|make a note)\b",
        ],
    ),
    # Identity, profile, and enduring personal facts (strength: 0.9)
    (
        "identity",
        0.9,
        [
            r"\b(my name is|i am a|i'm a|i live in|i speak|i work as|my job|my career|my role is)\b",
            r"\b(my email is|my phone is|i am originally from|i grew up in)\b",
        ],
    ),
    # Explicit preferences, goals, and passions (strength: 0.8)
    (
        "preference",
        0.8,
        [
            r"\bi (prefer|like|love|hate|dislike|enjoy|favor|favour|adore)\b",
            r"\b(favorite|favourite|interested in|passionate about)\b",
            r"\b(my goal is|my goal|i want to|i plan to|i hope to|i need to|i intend to)\b",
        ],
    ),
    # Active projects, research, engineering builds, and core preferences (strength: 0.85)
    (
        "project",
        0.85,
        [
            r"\b(i am working|i'm working|working on|i am doing|i'm doing|doing a project|project (of|on|about|is|called)|the project is|my project|i am building|i'm building|building a|i develop|developing|i study|i research|my research)\b",
            r"\b(i am learning|i'm learning|i learn)\b",
        ],
    ),
    # Collaborative and team/workflow context (strength: 0.5)
    (
        "team",
        0.5,
        [
            r"\b(we discussed|we decided|we agreed|we planned|team (discussed|decided|agreed|planned|assigned|reviewed))\b",
            r"\b(sprint|standup|retrospective|assigned tasks?|assigned to|teammates?|colleagues?|coworkers?)\b",
            r"\b(meeting|workshop|pair programming|code review|pull request|deployment|release|milestone)\b",
        ],
    ),
]

# Backward-compatibility flat pattern lists
INTENT_PATTERNS = [
    p for _, weight, patterns in INTENT_CATEGORY_PATTERNS if weight >= 0.7 for p in patterns
]
PARTIAL_INTENT_PATTERNS = [
    p for _, weight, patterns in INTENT_CATEGORY_PATTERNS if weight < 0.7 for p in patterns
]

# Specific domain keywords kept as a bonus signal
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

# English stop words / function words for information density calculation
FUNCTION_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
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
        breakdown = self.score_with_breakdown(content, access_count, created_at, now)
        return breakdown["total"]

    def score_with_breakdown(
        self,
        content: str,
        access_count: int = 0,
        created_at: datetime | None = None,
        now: datetime | None = None,
    ) -> dict:
        """Compute importance score with a detailed, interpretable signal breakdown."""
        filler_detected, filler_reason = is_filler(content)
        if filler_detected:
            return {
                "total": 0.0,
                "is_filler": True,
                "filler_reason": filler_reason,
                "signals": {},
            }

        text = content.lower().strip()
        words = text.split()
        if not words:
            return {
                "total": 0.0,
                "is_filler": True,
                "filler_reason": "empty",
                "signals": {},
            }

        weights = self.weights

        # 1. Compute Individual Signals & Rationales
        intent_score, intent_reason = self._compute_intent_score_and_reason(text)
        spec_score, spec_reason = self._compute_domain_agnostic_specificity(content, text, words)
        dur_score, dur_reason = self._compute_continuous_durability(text, intent_score)
        salience_score, salience_reason = self._compute_information_density(words, content)
        rec_score = min(max(access_count, 0) / 3.0, 1.0)
        recency_score = self._compute_recency(created_at, now)

        # 2. Weighted Sum
        signals = {
            "intent": {
                "value": round(intent_score, 4),
                "weight": weights.intent,
                "contribution": round(weights.intent * intent_score, 4),
                "reason": intent_reason,
            },
            "specificity": {
                "value": round(spec_score, 4),
                "weight": weights.specificity,
                "contribution": round(weights.specificity * spec_score, 4),
                "reason": spec_reason,
            },
            "durability": {
                "value": round(dur_score, 4),
                "weight": weights.durability,
                "contribution": round(weights.durability * dur_score, 4),
                "reason": dur_reason,
            },
            "salience": {
                "value": round(salience_score, 4),
                "weight": weights.salience,
                "contribution": round(weights.salience * salience_score, 4),
                "reason": salience_reason,
            },
            "recurrence": {
                "value": round(rec_score, 4),
                "weight": weights.recurrence,
                "contribution": round(weights.recurrence * rec_score, 4),
                "reason": f"Accessed {access_count} times",
            },
            "recency": {
                "value": round(recency_score, 4),
                "weight": weights.recency,
                "contribution": round(weights.recency * recency_score, 4),
                "reason": "Exponential age decay",
            },
        }

        total = sum(s["contribution"] for s in signals.values())
        bounded_total = max(0.0, min(1.0, round(total, 4)))

        return {
            "total": bounded_total,
            "is_filler": False,
            "filler_reason": "",
            "signals": signals,
        }

    @staticmethod
    def _compute_intent_score_and_reason(text: str) -> tuple[float, str]:
        """Graduated intent scoring based on intent category strength."""
        for category, strength, patterns in INTENT_CATEGORY_PATTERNS:
            for pat in patterns:
                m = re.search(pat, text)
                if m:
                    return strength, f"{category}: '{m.group(0)}'"
        return 0.0, "no explicit intent marker"

    @staticmethod
    def _compute_intent_score(text: str) -> float:
        score, _ = HeuristicImportanceScorer._compute_intent_score_and_reason(text)
        return score

    @staticmethod
    def _extract_entities(raw_text: str, lower_text: str) -> list[str]:
        """Lightweight entity extraction (acronyms, numbers, capitalized terms, quoted strings)."""
        entities = []
        # Acronyms (e.g. ADAM, DSA, LLM, API)
        acronyms = re.findall(r"\b[A-Z]{2,}\b", raw_text)
        entities.extend(acronyms)
        # Quoted terms
        quoted = re.findall(r'["\']([^"\']+)["\']', raw_text)
        entities.extend(quoted)
        # Numbers with units or standalone numbers
        numbers = re.findall(r"\b\d+(?:\.\d+)?(?:[a-zA-Z%]+)?\b", raw_text)
        entities.extend(numbers)
        # Capitalized proper nouns (excluding first word of sentence)
        tokens = raw_text.split()
        for i, token in enumerate(tokens[1:], start=1):
            clean = re.sub(r"[^\w]", "", token)
            if clean and clean[0].isupper() and clean.lower() not in FUNCTION_WORDS:
                entities.append(clean)
        return list(dict.fromkeys(entities))  # Deduplicate preserving order

    @staticmethod
    def _compute_domain_agnostic_specificity(
        raw_text: str, lower_text: str, words: list[str]
    ) -> tuple[float, str]:
        """Domain-agnostic specificity using entity density, vocab richness, and CS keywords bonus."""
        entities = HeuristicImportanceScorer._extract_entities(raw_text, lower_text)
        kw_matches = [kw for kw in SPECIFICITY_KEYWORDS if kw in lower_text]

        # Vocabulary richness: unique words ratio (excluding stop words)
        content_words = [w for w in words if w not in FUNCTION_WORDS]
        vocab_richness = len(set(content_words)) / max(len(content_words), 1)

        reasons = []
        score = 0.1

        # Base specificity from entity density and vocabulary richness
        if len(entities) >= 3 or (len(entities) >= 2 and vocab_richness > 0.7):
            score = max(score, 0.75)
            reasons.append(f"{len(entities)} entities detected ({', '.join(entities[:3])})")
        elif len(entities) >= 1:
            score = max(score, 0.45)
            reasons.append(f"Entity: {entities[0]}")
        elif len(content_words) >= 5 and vocab_richness > 0.8:
            score = max(score, 0.35)
            reasons.append(f"Rich content vocabulary ({len(content_words)} content words)")

        # CS / technical keywords bonus
        if len(kw_matches) >= 2 or (len(kw_matches) >= 1 and len(entities) >= 1):
            score = max(score, 1.0)
            reasons.append(f"Tech terms: {', '.join(kw_matches[:3])}")
        elif len(kw_matches) == 1:
            score = max(score, 0.70)
            reasons.append(f"Tech term: {kw_matches[0]}")

        reason_str = "; ".join(reasons) if reasons else "General conversational text"
        return min(1.0, score), reason_str

    @staticmethod
    def _compute_specificity_score(raw_text: str, lower_text: str, words: list[str]) -> float:
        score, _ = HeuristicImportanceScorer._compute_domain_agnostic_specificity(
            raw_text, lower_text, words
        )
        return score

    @staticmethod
    def _compute_continuous_durability(text: str, intent_score: float) -> tuple[float, str]:
        """Continuous durability score [0.0 - 1.0] based on temporal vs. permanent indicators."""
        # Check ephemeral signals
        ephemeral_match = next((pat for pat in EPHEMERAL_PATTERNS if re.search(pat, text)), None)
        if ephemeral_match:
            return 0.1, "ephemeral / transient context"

        # Architectural, permanent, or fundamental definitions
        durable_match = next((pat for pat in DURABLE_PATTERNS if re.search(pat, text)), None)
        if durable_match:
            return 0.9, "architectural / enduring definition"

        # Strong intent implies long-term durability
        if intent_score >= 0.8:
            return 0.85, "high intent implies enduring value"
        if intent_score >= 0.5:
            return 0.60, "moderate intent implies medium durability"

        # General factual sentences with copula ("X is Y")
        if re.search(r"\b(is a|are|consists|uses|requires)\b", text):
            return 0.50, "factual / descriptive statement"

        return 0.40, "default moderate durability"

    @staticmethod
    def _compute_durability_score(text: str, intent_score: float) -> float:
        score, _ = HeuristicImportanceScorer._compute_continuous_durability(text, intent_score)
        return score

    @staticmethod
    def _compute_information_density(words: list[str], raw_text: str) -> tuple[float, str]:
        """Information density and conversational salience rather than raw word count."""
        num_words = len(words)
        if num_words == 0:
            return 0.0, "empty"

        content_words = [w for w in words if w not in FUNCTION_WORDS]
        density_ratio = len(content_words) / num_words

        # Length factor: saturates around 20 words (instead of 15)
        length_factor = min(num_words / 20.0, 1.0)

        # Clause / structural complexity
        has_punctuation_structure = bool(re.search(r"[,;:—\n]", raw_text))
        complexity_bonus = 0.15 if has_punctuation_structure else 0.0

        salience = (0.50 * length_factor) + (0.35 * density_ratio) + complexity_bonus
        bounded = max(0.0, min(1.0, round(salience, 4)))
        return bounded, f"Density {density_ratio:.2f}, {len(content_words)}/{num_words} content words"

    @staticmethod
    def _compute_recency(created_at: datetime | None, now: datetime | None) -> float:
        if created_at is None:
            return 1.0
        current = now or datetime.now(timezone.utc)
        age_seconds = max(0.0, (current - created_at).total_seconds())
        half_life = 86400.0
        return 0.5 ** (age_seconds / half_life)