"""Environment-backed configuration for the Phase 1 service."""

import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "ADAM Phase 2"
    database_path: Path = Path("data/adam.db")
    embedding_model: str = "all-MiniLM-L6-v2"
    default_top_k: int = 5
    max_top_k: int = 20
    importance_persistent_weight: float = 0.55
    importance_length_weight: float = 0.20
    importance_recurrence_weight: float = 0.15
    importance_recency_weight: float = 0.10
    initial_archive_threshold: float = 0.20
    initial_long_term_threshold: float = 0.70
    working_to_long_term_age: timedelta = timedelta(days=7)
    working_to_long_term_min_accesses: int = 1
    short_term_to_archive_age: timedelta = timedelta(days=30)
    archive_compression_level: int = 1

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            database_path=Path(os.getenv("ADAM_DATABASE_PATH", "data/adam.db")),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            ),
            initial_archive_threshold=float(
                os.getenv("INITIAL_ARCHIVE_THRESHOLD", "0.20")
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
        )


settings = Settings.from_environment()