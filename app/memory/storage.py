"""SQLite persistence for the Phase 1 memory system."""

import sqlite3
import uuid
from pathlib import Path

from app.memory.models import Memory, utc_now


SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    importance_score REAL NOT NULL DEFAULT 0.0,
    tier TEXT NOT NULL DEFAULT 'WORKING'
)
"""


class SQLiteStorage:
    """Store memories in a local SQLite database."""

    def __init__(self, database_path: str | Path = "data/adam.db"):
        self.database_path = Path(database_path)
        self.initialize_database()

    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize_database(self) -> None:
        """Create the database directory and memories table if needed."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(SCHEMA)
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(memories)")
            }
            if "importance_score" not in columns:
                connection.execute(
                    "ALTER TABLE memories ADD COLUMN importance_score REAL NOT NULL DEFAULT 0.0"
                )
            if "tier" not in columns:
                connection.execute(
                    "ALTER TABLE memories ADD COLUMN tier TEXT NOT NULL DEFAULT 'WORKING'"
                )

    def save_memory(self, memory: Memory) -> Memory:
        """Persist one memory and return it unchanged."""
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO memories
                (memory_id, user_id, content, embedding, created_at,
                 last_accessed, access_count, importance_score, tier)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                memory.to_row(),
            )
        return memory

    def get_memories(self, user_id: str) -> list[Memory]:
        """Return all memories belonging to one user."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT memory_id, user_id, content, embedding, created_at,
                   last_accessed, access_count
                   , importance_score, tier
                   FROM memories WHERE user_id = ?""",
                (user_id,),
            ).fetchall()
        return [Memory.from_row(tuple(row)) for row in rows]

    def update_access_metadata(self, memory_id: str, accessed_at=None) -> None:
        """Update retrieval metadata for a returned memory."""
        accessed_at = accessed_at or utc_now()
        with self._connect() as connection:
            connection.execute(
                """UPDATE memories
                   SET last_accessed = ?, access_count = access_count + 1
                   WHERE memory_id = ?""",
                (accessed_at.isoformat(), memory_id),
            )

    def count(self) -> int:
        """Return the number of persisted memories."""
        with self._connect() as connection:
            return connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]


def create_memory(user_id: str, content: str, embedding: list[float]) -> Memory:
    now = utc_now()
    return Memory(
        memory_id=str(uuid.uuid4()),
        user_id=user_id,
        content=content,
        embedding=embedding,
        created_at=now,
        last_accessed=now,
    )
