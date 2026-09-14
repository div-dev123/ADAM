"""LLM-backed lifecycle compression transitions."""

from dataclasses import dataclass

from app.llm.client import LLMClient
from app.memory.models import Memory, utc_now
from app.memory.storage import SQLiteStorage
from app.memory.tiers import ARCHIVE, LONG_TERM, SHORT_TERM, WORKING


@dataclass(frozen=True)
class CompressionConfig:
    working_to_long_term_level: int = 1
    short_term_to_archive_level: int = 2


class CompressionService:
    """Compress only explicit lifecycle transitions, never normal creation."""

    def __init__(self, storage: SQLiteStorage, llm: LLMClient, embeddings,
                 config: CompressionConfig | None = None):
        self.storage = storage
        self.llm = llm
        self.embeddings = embeddings
        self.config = config or CompressionConfig()

    def transition(self, memory: Memory, target_tier: str) -> Memory:
        if memory.tier == target_tier:
            return memory

        level = self._level_for(memory.tier, target_tier)
        if level is None:
            raise ValueError(f"Unsupported compression transition: {memory.tier} -> {target_tier}")

        old_content = memory.content
        if level > 0:
            result = self.llm.compress_memory(memory.content, level)
            memory.content = result.compressed_content.strip()
            memory.embedding = self.embeddings.encode(memory.content)
            memory.compression_level = level
            reason = result.reason
            operation = "COMPRESSED"
        else:
            reason = f"Promoted/Reactivated to {target_tier}"
            operation = "PROMOTED"

        memory.tier = target_tier
        memory.updated_at = utc_now()
        self.storage.update_memory(memory)
        self.storage.record_history(memory, operation, old_content, reason)
        return memory

    def _level_for(self, source: str, target: str) -> int | None:
        if target == LONG_TERM:
            return self.config.working_to_long_term_level
        if target == ARCHIVE:
            return self.config.short_term_to_archive_level
        if target in {WORKING, SHORT_TERM}:
            return 0
        return None
