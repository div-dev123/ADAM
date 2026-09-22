"""Environment-backed configuration for the Phase 1 service."""

import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "ADAM Phase 3"
    database_path: Path = Path("data/adam.db")
    embedding_model: str = "all-MiniLM-L6-v2"
    default_top_k: int = 5
    max_top_k: int = 20
    importance_intent_weight: float = 0.35
    importance_specificity_weight: float = 0.25
    importance_durability_weight: float = 0.20
    importance_salience_weight: float = 0.10
    importance_recurrence_weight: float = 0.05
    importance_recency_weight: float = 0.05
    initial_archive_threshold: float = 0.30
    initial_long_term_threshold: float = 0.70
    working_to_long_term_age: timedelta = timedelta(days=7)
    working_to_long_term_min_accesses: int = 1
    short_term_to_archive_age: timedelta = timedelta(days=30)
    archive_compression_level: int = 1
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_timeout: float = 120.0
    consolidation_candidate_limit: int = 3
    consolidation_min_similarity: float = 0.35
    working_compression_level: int = 1
    archive_compression_level_target: int = 2
    search_mode: str = "hybrid"
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    # Phase 4 Lifecycle & Forgetting Settings
    lifecycle_forgetting_threshold: float = 0.75
    lifecycle_protected_importance: float = 0.70
    lifecycle_protected_access_count: int = 3
    lifecycle_working_age_days: float = 7.0
    lifecycle_short_term_age_days: float = 30.0
    lifecycle_archive_obsolete_days: float = 60.0
    lifecycle_recency_decay_rate: float = 0.05
    lifecycle_frequent_access_boost_threshold: int = 3
    # Phase 5 Adaptive Retrieval & Query Drift Settings
    drift_low_threshold: float = 0.55
    drift_high_threshold: float = 0.85
    drift_time_weight: float = 0.20
    drift_time_half_life_hours: float = 1.0

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            database_path=Path(os.getenv("ADAM_DATABASE_PATH", "data/adam.db")),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            ),
            importance_intent_weight=float(os.getenv("IMPORTANCE_INTENT_WEIGHT", "0.35")),
            importance_specificity_weight=float(os.getenv("IMPORTANCE_SPECIFICITY_WEIGHT", "0.25")),
            importance_durability_weight=float(os.getenv("IMPORTANCE_DURABILITY_WEIGHT", "0.20")),
            importance_salience_weight=float(os.getenv("IMPORTANCE_SALIENCE_WEIGHT", "0.10")),
            importance_recurrence_weight=float(os.getenv("IMPORTANCE_RECURRENCE_WEIGHT", "0.05")),
            importance_recency_weight=float(os.getenv("IMPORTANCE_RECENCY_WEIGHT", "0.05")),
            initial_archive_threshold=float(
                os.getenv("INITIAL_ARCHIVE_THRESHOLD", "0.30")
            ),
            initial_long_term_threshold=float(os.getenv("INITIAL_LONG_TERM_THRESHOLD", "0.70")),
            working_to_long_term_age=timedelta(
                days=float(os.getenv("WORKING_TO_LONG_TERM_DAYS", "7"))
            ),
            working_to_long_term_min_accesses=int(
                os.getenv("WORKING_TO_LONG_TERM_MIN_ACCESSES", "1")
            ),
            short_term_to_archive_age=timedelta(
                days=float(os.getenv("SHORT_TERM_TO_ARCHIVE_DAYS", "30"))
            ),
            archive_compression_level=int(
                os.getenv("ARCHIVE_COMPRESSION_LEVEL", "1")
            ),
            ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
            ollama_timeout=float(os.getenv("OLLAMA_TIMEOUT", "120.0")),
            consolidation_candidate_limit=int(
                os.getenv("CONSOLIDATION_CANDIDATE_LIMIT", "3")
            ),
            consolidation_min_similarity=float(
                os.getenv("CONSOLIDATION_MIN_SIMILARITY", "0.35")
            ),
            working_compression_level=int(
                os.getenv("WORKING_COMPRESSION_LEVEL", "1")
            ),
            archive_compression_level_target=int(
                os.getenv("ARCHIVE_COMPRESSION_LEVEL_TARGET", "2")
            ),
            search_mode=os.getenv("SEARCH_MODE", "hybrid"),
            rrf_k=int(os.getenv("RRF_K", "60")),
            bm25_k1=float(os.getenv("BM25_K1", "1.5")),
            bm25_b=float(os.getenv("BM25_B", "0.75")),
            lifecycle_forgetting_threshold=float(
                os.getenv("LIFECYCLE_FORGETTING_THRESHOLD", "0.75")
            ),
            lifecycle_protected_importance=float(
                os.getenv("LIFECYCLE_PROTECTED_IMPORTANCE", "0.70")
            ),
            lifecycle_protected_access_count=int(
                os.getenv("LIFECYCLE_PROTECTED_ACCESS_COUNT", "3")
            ),
            lifecycle_working_age_days=float(
                os.getenv("LIFECYCLE_WORKING_AGE_DAYS", "7.0")
            ),
            lifecycle_short_term_age_days=float(
                os.getenv("LIFECYCLE_SHORT_TERM_AGE_DAYS", "30.0")
            ),
            lifecycle_archive_obsolete_days=float(
                os.getenv("LIFECYCLE_ARCHIVE_OBSOLETE_DAYS", "60.0")
            ),
            lifecycle_recency_decay_rate=float(
                os.getenv("LIFECYCLE_RECENCY_DECAY_RATE", "0.05")
            ),
            lifecycle_frequent_access_boost_threshold=int(
                os.getenv("LIFECYCLE_FREQUENT_ACCESS_BOOST_THRESHOLD", "3")
            ),
            drift_low_threshold=float(
                os.getenv("DRIFT_LOW_THRESHOLD", "0.55")
            ),
            drift_high_threshold=float(
                os.getenv("DRIFT_HIGH_THRESHOLD", "0.85")
            ),
            drift_time_weight=float(
                os.getenv("DRIFT_TIME_WEIGHT", "0.20")
            ),
            drift_time_half_life_hours=float(
                os.getenv("DRIFT_TIME_HALF_LIFE_HOURS", "1.0")
            ),
        )


settings = Settings.from_environment()