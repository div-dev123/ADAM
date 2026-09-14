"""Environment-backed configuration for the Phase 1 service."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "ADAM Phase 1"
    database_path: Path = Path("data/adam.db")
    embedding_model: str = "all-MiniLM-L6-v2"
    default_top_k: int = 5
    max_top_k: int = 20

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            database_path=Path(os.getenv("ADAM_DATABASE_PATH", "data/adam.db")),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            ),
        )


settings = Settings.from_environment()