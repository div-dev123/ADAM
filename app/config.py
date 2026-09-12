"""Environment-backed configuration for the Phase 1 service."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "ADAM Phase 1"
    mongo_uri: str | None = None
    mongo_database: str = "adam_memory"
    mongo_collection: str = "memories"
    embedding_model: str = "all-MiniLM-L6-v2"
    default_top_k: int = 5
    max_top_k: int = 20

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            mongo_uri=os.getenv("MONGODB_URI"),
            mongo_database=os.getenv("MONGODB_DATABASE", "adam_memory"),
            mongo_collection=os.getenv("MONGODB_COLLECTION", "memories"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            ),
        )


settings = Settings.from_environment()