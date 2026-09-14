"""SQLite persistence for the Phase 1 memory system."""

import sqlite3
import uuid
import json
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
    tier TEXT NOT NULL DEFAULT 'WORKING',
    compression_level INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS memory_history (
    history_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    old_content TEXT,
    new_content TEXT,
    reason TEXT,
    created_at TEXT NOT NULL
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
            connection.executescript(SCHEMA)
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
            if "compression_level" not in columns:
                connection.execute(
                    "ALTER TABLE memories ADD COLUMN compression_level INTEGER NOT NULL DEFAULT 0"
                )
            if "updated_at" not in columns:
                connection.execute(
                    "ALTER TABLE memories ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                "UPDATE memories SET updated_at = created_at WHERE updated_at = ''"
            )

    def save_memory(self, memory: Memory) -> Memory:
        """Persist one memory and return it unchanged."""
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO memories
                (memory_id, user_id, content, embedding, created_at,
                 last_accessed, access_count, importance_score, tier,
                 compression_level, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                memory.to_row(),
            )
        return memory

    def get_memories(self, user_id: str) -> list[Memory]:
        """Return all memories belonging to one user."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT memory_id, user_id, content, embedding, created_at,
                   last_accessed, access_count
                   , importance_score, tier, compression_level, updated_at
                   FROM memories WHERE user_id = ?""",
                (user_id,),
            ).fetchall()
        return [Memory.from_row(tuple(row)) for row in rows]

    def get_memory(self, memory_id: str) -> Memory | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT memory_id, user_id, content, embedding, created_at,
                   last_accessed, access_count, importance_score, tier,
                   compression_level, updated_at
                   FROM memories WHERE memory_id = ?""",
                (memory_id,),
            ).fetchone()
        return Memory.from_row(tuple(row)) if row else None

    def update_access_metadata(self, memory_id: str, accessed_at=None) -> None:
        """Update retrieval metadata for a returned memory."""
        accessed_at = accessed_at or utc_now()
        with self._connect() as connection:
            connection.execute(
                     """UPDATE memories
                         SET last_accessed = ?, updated_at = ?,
                              access_count = access_count + 1
                   WHERE memory_id = ?""",
                     (accessed_at.isoformat(), accessed_at.isoformat(), memory_id),
            )

    def update_memory(self, memory: Memory) -> Memory:
        """Persist an updated consolidated memory."""
        with self._connect() as connection:
            connection.execute(
                """UPDATE memories SET content = ?, embedding = ?,
                   last_accessed = ?, access_count = ?, importance_score = ?,
                   tier = ?, compression_level = ?, updated_at = ?
                   WHERE memory_id = ?""",
                (
                    memory.content,
                    json.dumps(memory.embedding),
                    memory.last_accessed.isoformat(),
                    memory.access_count,
                    memory.importance_score,
                    memory.tier,
                    memory.compression_level,
                    memory.updated_at.isoformat(),
                    memory.memory_id,
                ),
            )
        return memory

    def record_history(
        self, memory: Memory, operation: str, old_content: str | None,
        reason: str = "",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO memory_history
                   (history_id, memory_id, user_id, operation, old_content,
                    new_content, reason, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()), memory.memory_id, memory.user_id,
                    operation, old_content, memory.content, reason,
                    utc_now().isoformat(),
                ),
            )

    def get_history(self, memory_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_history WHERE memory_id = ? ORDER BY created_at",
                (memory_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        """Return the number of persisted memories."""
        with self._connect() as connection:
            return connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def get_all_memories(
        self,
        user_id: str | None = None,
        tier: str | None = None,
        search: str | None = None,
    ) -> list[Memory]:
        """Fetch memories with optional user, tier, or search filters."""
        query = """SELECT memory_id, user_id, content, embedding, created_at,
                          last_accessed, access_count, importance_score, tier,
                          compression_level, updated_at
                   FROM memories WHERE 1=1"""
        params = []
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if tier:
            query += " AND tier = ?"
            params.append(tier.upper())
        if search:
            query += " AND content LIKE ?"
            params.append(f"%{search}%")
        query += " ORDER BY datetime(created_at) DESC"

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [Memory.from_row(tuple(row)) for row in rows]

    def get_metrics(self, user_id: str | None = None) -> dict:
        """Calculate live research metrics from storage and history."""
        user_filter = "WHERE user_id = ?" if user_id else ""
        user_param = (user_id,) if user_id else ()

        with self._connect() as connection:
            total_memories = connection.execute(
                f"SELECT COUNT(*) FROM memories {user_filter}", user_param
            ).fetchone()[0]

            tier_rows = connection.execute(
                f"SELECT tier, COUNT(*) FROM memories {user_filter} GROUP BY tier",
                user_param,
            ).fetchall()
            tier_counts = {
                "WORKING": 0,
                "SHORT_TERM": 0,
                "LONG_TERM": 0,
                "ARCHIVE": 0,
            }
            for row in tier_rows:
                tier_counts[row[0]] = row[1]

            avg_importance_row = connection.execute(
                f"SELECT AVG(importance_score) FROM memories {user_filter}",
                user_param,
            ).fetchone()
            avg_importance = round(avg_importance_row[0], 4) if avg_importance_row and avg_importance_row[0] is not None else 0.0

            compressed_count = connection.execute(
                f"SELECT COUNT(*) FROM memories WHERE compression_level > 0 {'AND user_id = ?' if user_id else ''}",
                user_param,
            ).fetchone()[0]

            history_filter = "WHERE user_id = ?" if user_id else ""
            consolidation_rows = connection.execute(
                f"SELECT operation, COUNT(*) FROM memory_history {history_filter} GROUP BY operation",
                user_param,
            ).fetchall()
            consolidation_counts = {
                "DUPLICATE": 0,
                "RELATED": 0,
                "CONTRADICTORY": 0,
                "COMPRESSED": 0,
            }
            for row in consolidation_rows:
                consolidation_counts[row[0]] = row[1]

        return {
            "total_memories": total_memories,
            "tier_counts": tier_counts,
            "avg_importance": avg_importance,
            "compressed_memories": compressed_count,
            "archived_memories": tier_counts.get("ARCHIVE", 0),
            "consolidation_counts": consolidation_counts,
            "total_consolidations": sum(
                consolidation_counts.get(k, 0)
                for k in ("DUPLICATE", "RELATED", "CONTRADICTORY")
            ),
        }

    def get_all_history(self, limit: int = 50, user_id: str | None = None) -> list[dict]:
        """Fetch the most recent consolidation/compression history records."""
        query = "SELECT * FROM memory_history"
        params = []
        if user_id:
            query += " WHERE user_id = ?"
            params.append(user_id)
        query += " ORDER BY datetime(created_at) DESC LIMIT ?"
        params.append(limit)

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a single memory and its history."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM memories WHERE memory_id = ?", (memory_id,)
            )
            connection.execute(
                "DELETE FROM memory_history WHERE memory_id = ?", (memory_id,)
            )
            return cursor.rowcount > 0

    def reset_database(self) -> None:
        """Truncate all memories and history tables."""
        with self._connect() as connection:
            connection.execute("DELETE FROM memories")
            connection.execute("DELETE FROM memory_history")


def create_memory(user_id: str, content: str, embedding: list[float]) -> Memory:
    now = utc_now()
    return Memory(
        memory_id=str(uuid.uuid4()),
        user_id=user_id,
        content=content,
        embedding=embedding,
        created_at=now,
        last_accessed=now,
        updated_at=now,
    )
