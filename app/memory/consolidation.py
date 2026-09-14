"""LLM-assisted consolidation for the Phase 3 write path."""

from dataclasses import dataclass

from app.llm.client import LLMClient
from app.memory.importance import is_filler
from app.memory.models import Memory, utc_now
from app.memory.storage import SQLiteStorage, create_memory
from app.retrieval.similarity import cosine_similarity


@dataclass(frozen=True)
class ConsolidationConfig:
    candidate_limit: int = 3
    min_similarity: float = 0.35


class ConsolidationService:
    def __init__(self, storage: SQLiteStorage, llm: LLMClient, embeddings,
                 scorer, lifecycle_policy, config: ConsolidationConfig):
        self.storage = storage
        self.llm = llm
        self.embeddings = embeddings
        self.scorer = scorer
        self.lifecycle_policy = lifecycle_policy
        self.config = config

    def process(self, user_id: str, content: str) -> Memory | None:
        trace = self.process_with_trace(user_id, content)
        return trace.get("memory")

    def process_with_trace(self, user_id: str, content: str) -> dict:
        filler_detected, filler_reason = is_filler(content)
        if filler_detected:
            return {
                "memory": None,
                "action": "FILLER",
                "decision_reason": f"Filtered greeting or conversational filler ({filler_reason})",
                "merged_content": None,
                "old_content": None,
                "candidates": [],
                "is_stored": False,
                "importance_score": 0.0,
                "tier": None,
                "compression_level": 0,
            }

        embedding = self.embeddings.encode(content)
        candidates = self._candidates(user_id, embedding)
        candidates_info = [
            {
                "memory_id": memory.memory_id,
                "content": memory.content,
                "similarity": round(float(score), 4),
                "tier": memory.tier,
                "importance_score": memory.importance_score,
            }
            for score, memory in candidates
        ]

        try:
            decision = self.llm.classify_memory(content, [
                {"memory_id": c["memory_id"], "content": c["content"], "similarity": c["similarity"]}
                for c in candidates_info
            ])
        except Exception as error:
            # Fallback to NEW when LLM classification is unavailable
            decision = type("FallbackDecision", (), {
                "action": "NEW",
                "reason": f"Fallback to NEW (LLM unavailable: {error})",
                "merged_content": None,
                "merged_text": lambda self: "",
            })()

        target = candidates[0][1] if candidates else None
        old_content = target.content if target else None

        if decision.action == "DUPLICATE" and target:
            accessed_at = utc_now()
            self.storage.update_access_metadata(target.memory_id, accessed_at)
            target.last_accessed = accessed_at
            target.updated_at = accessed_at
            target.access_count += 1
            self.storage.record_history(target, "DUPLICATE", target.content, decision.reason)
            memory = target
        elif decision.action in {"RELATED", "CONTRADICTORY"} and target:
            memory = self._update(target, content, decision)
        else:
            memory = self._create(user_id, content, embedding)

        return {
            "memory": memory,
            "action": decision.action,
            "decision_reason": getattr(decision, "reason", ""),
            "merged_content": getattr(decision, "merged_content", None),
            "old_content": old_content if decision.action != "NEW" else None,
            "candidates": candidates_info,
            "is_stored": True,
            "importance_score": memory.importance_score,
            "tier": memory.tier,
            "compression_level": memory.compression_level,
        }

    def _candidates(self, user_id: str, embedding):
        ranked = sorted(
            ((cosine_similarity(embedding, memory.embedding), memory)
             for memory in self.storage.get_memories(user_id)),
            key=lambda item: item[0], reverse=True,
        )
        return [item for item in ranked if item[0] >= self.config.min_similarity][
            : self.config.candidate_limit
        ]

    def _create(self, user_id, content, embedding):
        memory = create_memory(user_id, content, embedding)
        self._classify(memory)
        self.storage.save_memory(memory)
        return memory

    def _update(self, target, incoming_content, decision):
        old_content = target.content
        target.content = decision.merged_text()
        target.embedding = self.embeddings.encode(target.content)
        target.importance_score = self.scorer.score(
            target.content, target.access_count, target.created_at
        )
        target.access_count += 1
        target.last_accessed = utc_now()
        target.updated_at = target.last_accessed
        self.storage.update_memory(target)
        self.storage.record_history(target, decision.action, old_content, decision.reason)
        return target

    def _classify(self, memory):
        memory.importance_score = self.scorer.score(
            memory.content, memory.access_count, memory.created_at
        )
        memory.tier = self.lifecycle_policy.initial_tier(memory.importance_score)

