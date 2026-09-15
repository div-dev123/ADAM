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
    consolidation_min_similarity: float = 0.60
    working_compression_level: int = 1
    archive_compression_level_target: int = 2

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
                os.getenv("CONSOLIDATION_MIN_SIMILARITY", "0.60")
            ),
            working_compression_level=int(
                os.getenv("WORKING_COMPRESSION_LEVEL", "1")
            ),
            archive_compression_level_target=int(
                os.getenv("ARCHIVE_COMPRESSION_LEVEL_TARGET", "2")
            ),
        )


settings = Settings.from_environment()