"""Phase 1 semantic retrieval service."""

from app.memory.models import Memory, utc_now
from app.memory.compression import CompressionService
from app.memory.consolidation import ConsolidationConfig, ConsolidationService
from app.memory.importance import HeuristicImportanceScorer, ImportanceWeights
from app.memory.storage import SQLiteStorage, create_memory
from app.memory.tiers import TierAssigner
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.similarity import cosine_similarity


class RetrievalService:
    def __init__(
        self,
        storage: SQLiteStorage,
        embeddings: EmbeddingService,
        scorer=None,
        tier_assigner=None,
        lifecycle_policy=None,
        llm=None,
        consolidation_config: ConsolidationConfig | None = None,
        compression_config=None,
    ):
        self.storage = storage
        self.embeddings = embeddings
        self.scorer = scorer or HeuristicImportanceScorer(ImportanceWeights())
        self.tier_assigner = lifecycle_policy or tier_assigner or TierAssigner()
        self.consolidation = (
            ConsolidationService(
                storage, llm, embeddings, self.scorer, self.tier_assigner,
                consolidation_config or ConsolidationConfig(),
            )
            if llm else None
        )
        self.compression = (
            CompressionService(storage, llm, embeddings, compression_config)
            if llm else None
        )

    def store_memory(self, user_id: str, content: str) -> Memory:
        if self.consolidation:
            return self.consolidation.process(user_id, content)
        memory = create_memory(user_id, content, self.embeddings.encode(content))
        memory.importance_score = self.scorer.score(
            memory.content,
            access_count=memory.access_count,
            created_at=memory.created_at,
        )
        memory.tier = self.tier_assigner.initial_tier(memory.importance_score)
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
            memory.updated_at = accessed_at
            memory.access_count += 1
            results.append({"memory": memory, "similarity": score})
        return results

    def transition_memory(self, memory_id: str, target_tier: str) -> Memory:
        """Apply one explicit compressed lifecycle transition."""
        if not self.compression:
            raise RuntimeError("LLM compression is not configured")
        memory = self.storage.get_memory(memory_id)
        if memory is None:
            raise KeyError(memory_id)
        return self.compression.transition(memory, target_tier)