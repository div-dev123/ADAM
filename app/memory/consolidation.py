"""LLM-assisted memory consolidation for Phase 3."""

from dataclasses import dataclass

from app.llm.client import ConsolidationDecision, LLMClient
from app.memory.models import Memory, utc_now
from app.memory.storage import SQLiteStorage, create_memory
from app.retrieval.similarity import cosine_similarity


@dataclass(frozen=True)
class ConsolidationConfig:
    candidate_limit: int = 3
    min_similarity: float = 0.35


class ConsolidationService:
    """Analyze only semantically close memories before writing a new memory."""

    def __init__(
        self,
        storage: SQLiteStorage,
        llm: LLMClient,
        embeddings,
        scorer,
        tier_assigner,
        config: ConsolidationConfig | None = None,
    ):
        self.storage = storage
        self.llm = llm
        self.embeddings = embeddings
        self.scorer = scorer
        self.tier_assigner = tier_assigner
        self.config = config or ConsolidationConfig()

    def process(self, user_id: str, content: str) -> Memory:
        embedding = self.embeddings.encode(content)
        candidates = self._candidate_memories(user_id, embedding)
        decision = self.llm.analyze_consolidation(
            content,
            [
                {
                    "memory_id": memory.memory_id,
                    "content": memory.content,
                    "similarity": similarity,
                }
                for similarity, memory in candidates
            ],
        )
        target = candidates[0][1] if candidates else None
        if decision.action == "DUPLICATE" and target:
            accessed_at = utc_now()
            self.storage.update_access_metadata(target.memory_id, accessed_at)
            target.last_accessed = accessed_at
            target.access_count += 1
            self.storage.record_consolidation_event(
                user_id, content, target, decision
            )
            return target
        if decision.action in {"RELATED", "CONTRADICTORY"} and target:
            return self._update_existing(target, content, decision)
        return self._save_new(user_id, content, embedding)

    def _candidate_memories(self, user_id: str, embedding: list[float]):
        ranked = [
            (cosine_similarity(embedding, memory.embedding), memory)
            for memory in self.storage.get_memories(user_id)
        ]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            item for item in ranked
            if item[0] >= self.config.min_similarity
        ][: self.config.candidate_limit]

    def _save_new(self, user_id: str, content: str, embedding: list[float]):
        memory = create_memory(user_id, content, embedding)
        self._classify(memory)
        self.storage.save_memory(memory)
        return memory

    def _update_existing(
        self, target: Memory, new_content: str, decision: ConsolidationDecision
    ):
        merged_content = decision.require_merged_content()
        old_content = target.content
        target.content = merged_content
        target.embedding = self.embeddings.encode(merged_content)
        target.importance_score = self.scorer.score(
            merged_content,
            access_count=target.access_count,
            created_at=target.created_at,
        )
        target.tier = self.tier_assigner.assign(
            target.importance_score, target.access_count
        )
        target.last_accessed = utc_now()
        target.access_count += 1
        self.storage.update_memory(target)
        self.storage.record_consolidation_event(
            target.user_id,
            new_content,
            target,
            decision,
            old_content=old_content,
        )
        return target

    def _classify(self, memory: Memory):
        memory.importance_score = self.scorer.score(
            memory.content,
            access_count=memory.access_count,
            created_at=memory.created_at,
        )
        memory.tier = self.tier_assigner.assign(
            memory.importance_score, memory.access_count
        )
