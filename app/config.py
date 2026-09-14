"""Environment-backed configuration for the Phase 1 service."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "ADAM Phase 3"
    database_path: Path = Path("data/adam.db")
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"
    consolidation_candidate_limit: int = 3
    consolidation_min_similarity: float = 0.35
    default_top_k: int = 5
    max_top_k: int = 20
    importance_persistent_weight: float = 0.55
    importance_length_weight: float = 0.20
    importance_recurrence_weight: float = 0.15
    importance_recency_weight: float = 0.10
    archive_threshold: float = 0.20
    working_threshold: float = 0.45
    short_term_threshold: float = 0.70

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            database_path=Path(os.getenv("ADAM_DATABASE_PATH", "data/adam.db")),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            ),
            ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
            consolidation_candidate_limit=int(
                os.getenv("CONSOLIDATION_CANDIDATE_LIMIT", "3")
            ),
            consolidation_min_similarity=float(
                os.getenv("CONSOLIDATION_MIN_SIMILARITY", "0.35")
            ),
        )


settings = Settings.from_environment()