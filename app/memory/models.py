"""Minimal Phase 1 memory data model."""

from dataclasses import dataclass
from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Memory:
    memory_id: str
    user_id: str
    content: str
    embedding: list[float]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0

    def to_document(self) -> dict:
        return {
            "_id": self.memory_id,
            "user_id": self.user_id,
            "content": self.content,
            "embedding": self.embedding,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
        }

    @classmethod
    def from_document(cls, document: dict) -> "Memory":
        return cls(
            memory_id=str(document["_id"]),
            user_id=document["user_id"],
            content=document["content"],
            embedding=list(document["embedding"]),
            created_at=document["created_at"],
            last_accessed=document["last_accessed"],
            access_count=document.get("access_count", 0),
        )