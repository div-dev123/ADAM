"""Minimal Phase 1 memory data model."""

import json
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

    def to_row(self) -> tuple:
        return (
            self.memory_id,
            self.user_id,
            self.content,
            json.dumps(self.embedding),
            self.created_at.isoformat(),
            self.last_accessed.isoformat(),
            self.access_count,
        )

    @classmethod
    def from_row(cls, row: tuple) -> "Memory":
        return cls(
            memory_id=row[0],
            user_id=row[1],
            content=row[2],
            embedding=json.loads(row[3]),
            created_at=datetime.fromisoformat(row[4]),
            last_accessed=datetime.fromisoformat(row[5]),
            access_count=row[6],
        )