"""Phase 1 semantic retrieval service."""

import math

from app.memory.models import Memory, utc_now
from app.memory.importance import HeuristicImportanceScorer, ImportanceWeights
from app.memory.storage import SQLiteStorage, create_memory
from app.memory.tiers import TierAssigner, TierThresholds
from app.retrieval.embeddings import EmbeddingService


class RetrievalService:
    def __init__(self, storage: SQLiteStorage, embeddings: EmbeddingService,
                 scorer=None, tier_assigner=None):
        self.storage = storage
        self.embeddings = embeddings
        self.scorer = scorer or HeuristicImportanceScorer(ImportanceWeights())
        self.tier_assigner = tier_assigner or TierAssigner(TierThresholds())

    def store_memory(self, user_id: str, content: str) -> Memory:
        memory = create_memory(user_id, content, self.embeddings.encode(content))
        memory.importance_score = self.scorer.score(
            memory.content,
            access_count=memory.access_count,
            created_at=memory.created_at,
        )
        memory.tier = self.tier_assigner.assign(
            memory.importance_score,
            access_count=memory.access_count,
        )
        return self.storage.save_memory(memory)

    def search(self, user_id: str, query: str, top_k: int):
        query_embedding = self.embeddings.encode(query)
        memories = self.storage.get_memories(user_id)
        ranked = sorted(
            (
                (cosine_similarity(query_embedding, memory.embedding), memory)
                for memory in memories
            ),
            key=lambda item: item[0],
            reverse=True,
        )[:top_k]
        results = []
        for score, memory in ranked:
            accessed_at = utc_now()
            self.storage.update_access_metadata(memory.memory_id, accessed_at)
            memory.last_accessed = accessed_at
            memory.access_count += 1
            results.append({"memory": memory, "similarity": score})
        return results


def cosine_similarity(first: list[float], second: list[float]) -> float:
    """Return cosine similarity without adding ranking signals."""
    if not first or not second or len(first) != len(second):
        return 0.0
    dot_product = sum(left * right for left, right in zip(first, second))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if not first_norm or not second_norm:
        return 0.0
    return dot_product / (first_norm * second_norm)