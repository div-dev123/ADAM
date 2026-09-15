"""Memory data model with intrinsic value and lifecycle state separated."""

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
    importance_score: float = 0.0
    tier: str = "WORKING"
    compression_level: int = 0
    updated_at: datetime | None = None
    superseded_by: str | None = None

    def __post_init__(self):
        if self.updated_at is None:
            self.updated_at = self.created_at

    def to_row(self) -> tuple:
        return (
            self.memory_id,
            self.user_id,
            self.content,
            json.dumps(self.embedding),
            self.created_at.isoformat(),
            self.last_accessed.isoformat(),
            self.access_count,
            self.importance_score,
            self.tier,
            self.compression_level,
            self.updated_at.isoformat(),
            self.superseded_by or "",
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
            importance_score=row[7],
            tier=row[8],
            compression_level=row[9] if len(row) > 9 else 0,
            updated_at=(
                datetime.fromisoformat(row[10])
                if len(row) > 10 and row[10]
                else datetime.fromisoformat(row[4])
            ),
            superseded_by=row[11] if len(row) > 11 and row[11] else None,
        )