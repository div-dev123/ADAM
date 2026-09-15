"""Phase 1 semantic retrieval service with dual-turn memory storage."""

from app.memory.models import Memory, utc_now
from app.memory.compression import CompressionService
from app.memory.consolidation import ConsolidationConfig, ConsolidationService
from app.memory.importance import HeuristicImportanceScorer, ImportanceWeights, is_filler
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
        self.llm = llm
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

    def store_memory(self, user_id: str, content: str, source_role: str = "user") -> Memory | None:
        trace = self.store_memory_with_trace(user_id, content, source_role=source_role)
        return trace.get("memory")

    def store_memory_with_trace(self, user_id: str, content: str, source_role: str = "user") -> dict:
        """Store memory and return complete decision metadata for research inspection."""
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
                "score_breakdown": {},
            }

        score_breakdown = (
            self.scorer.score_with_breakdown(content)
            if hasattr(self.scorer, "score_with_breakdown")
            else {"total": self.scorer.score(content), "signals": {}}
        )

        if self.consolidation:
            return self.consolidation.process_with_trace(user_id, content, source_role=source_role)

        memory = create_memory(user_id, content, self.embeddings.encode(content))
        memory.importance_score = score_breakdown.get("total", self.scorer.score(
            memory.content,
            access_count=memory.access_count,
            created_at=memory.created_at,
        ))
        memory.tier = self.tier_assigner.initial_tier(memory.importance_score)
        self.storage.save_memory(memory)
        return {
            "memory": memory,
            "action": "NEW",
            "decision_reason": "Default heuristic write (no LLM consolidation)",
            "merged_content": None,
            "old_content": None,
            "candidates": [],
            "is_stored": True,
            "importance_score": memory.importance_score,
            "tier": memory.tier,
            "compression_level": memory.compression_level,
            "score_breakdown": score_breakdown.get("signals", {}),
        }

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

    def chat_turn(
        self,
        user_id: str,
        message: str,
        top_k: int = 5,
        chat_history: list[dict] | None = None,
    ) -> dict:
        """Execute a full conversational turn with transparent memory processing for both user and LLM."""
        pipeline_stages = []

        # 1. User Message Memory Processing (Extraction, Importance, Tier, Consolidation)
        user_memory_trace = self.store_memory_with_trace(user_id, message, source_role="user")
        if user_memory_trace["is_stored"]:
            pipeline_stages.append({
                "stage": "User Message Memory Analysis & Scoring",
                "status": "completed",
                "detail": f"Importance: {user_memory_trace['importance_score']:.2f} (Tier: {user_memory_trace['tier']})",
            })
            pipeline_stages.append({
                "stage": "User Consolidation Check",
                "status": "completed",
                "detail": f"Action: {user_memory_trace['action']} | Candidates: {len(user_memory_trace['candidates'])}",
            })
        else:
            pipeline_stages.append({
                "stage": "User Message Memory Analysis",
                "status": "completed",
                "detail": f"Ignored greeting / filler ({user_memory_trace['decision_reason']})",
            })

        # 2. Semantic Memory Retrieval for User Query
        retrieved_raw = self.search(user_id, message, top_k=top_k)
        pipeline_stages.append({
            "stage": "Memory Retrieval",
            "status": "completed",
            "detail": f"Retrieved {len(retrieved_raw)} relevant memories",
        })

        # 3. Context Assembly & LLM Response Generation
        pipeline_stages.append({
            "stage": "Context Assembly & LLM Generation",
            "status": "in_progress",
            "detail": "Assembling context and invoking LLM...",
        })

        response_text = ""
        llm_error = None
        if self.llm and hasattr(self.llm, "generate_chat_response"):
            try:
                response_text = self.llm.generate_chat_response(
                    user_message=message,
                    retrieved_memories=retrieved_raw,
                    chat_history=chat_history,
                )
            except Exception as error:
                llm_error = str(error)
                response_text = (
                    f"I received your message and processed it into ADAM memory. "
                    f"(Note: Local Ollama generation was unavailable: {error})"
                )
        else:
            response_text = (
                "Memory processed successfully in ADAM. (Ollama client not configured for chat generation)"
            )

        pipeline_stages[-1]["status"] = "completed"
        pipeline_stages[-1]["detail"] = f"Generated {len(response_text)} chars"

        # 4. Assistant Response Memory Processing (Analysis & Storage)
        assistant_is_filler, filler_reason = is_filler(response_text)
        if llm_error:
            # Fallback error notices must never be scored or stored as memory
            assistant_memory_trace = {
                "memory": None,
                "action": "ERROR_FALLBACK",
                "decision_reason": f"Ollama generation failed or timed out ({llm_error}); error notice not stored in memory",
                "merged_content": None,
                "old_content": None,
                "candidates": [],
                "is_stored": False,
                "importance_score": 0.0,
                "tier": None,
                "compression_level": 0,
                "score_breakdown": {},
            }
            pipeline_stages.append({
                "stage": "Response Memory Processing",
                "status": "completed",
                "detail": f"Ollama unavailable; omitted error message from memory storage",
            })
        elif assistant_is_filler:
            assistant_memory_trace = {
                "memory": None,
                "action": "FILLER",
                "decision_reason": f"Filtered assistant boilerplate or greeting ({filler_reason})",
                "merged_content": None,
                "old_content": None,
                "candidates": [],
                "is_stored": False,
                "importance_score": 0.0,
                "tier": None,
                "compression_level": 0,
                "score_breakdown": {},
            }
            pipeline_stages.append({
                "stage": "Response Memory Processing",
                "status": "completed",
                "detail": f"Ignored boilerplate response ({filler_reason})",
            })
        else:
            # Informative LLM response is analyzed and stored in memory with role isolation
            assistant_memory_trace = self.store_memory_with_trace(user_id, response_text, source_role="assistant")
            pipeline_stages.append({
                "stage": "Response Memory Processing",
                "status": "completed",
                "detail": f"Stored Response Memory: {assistant_memory_trace['importance_score']:.2f} (Tier: {assistant_memory_trace['tier']}, Action: {assistant_memory_trace['action']})",
            })

        def memory_dict(mem):
            if mem is None:
                return None
            return {
                "memory_id": mem.memory_id,
                "user_id": mem.user_id,
                "content": mem.content,
                "importance_score": mem.importance_score,
                "tier": mem.tier,
                "compression_level": mem.compression_level,
                "created_at": mem.created_at.isoformat(),
                "last_accessed": mem.last_accessed.isoformat(),
                "access_count": mem.access_count,
                "updated_at": mem.updated_at.isoformat() if mem.updated_at else mem.created_at.isoformat(),
                "superseded_by": getattr(mem, "superseded_by", None),
            }

        user_mem_obj = user_memory_trace.get("memory")
        user_mem_data = memory_dict(user_mem_obj) if user_mem_obj else None

        assistant_mem_obj = assistant_memory_trace.get("memory")
        assistant_mem_data = memory_dict(assistant_mem_obj) if assistant_mem_obj else None

        return {
            "response": response_text,
            "user_memory": {
                "memory": user_mem_data,
                "action": user_memory_trace.get("action", "NONE"),
                "decision_reason": user_memory_trace.get("decision_reason", ""),
                "merged_content": user_memory_trace.get("merged_content"),
                "old_content": user_memory_trace.get("old_content"),
                "candidates": user_memory_trace.get("candidates", []),
                "is_stored": user_memory_trace.get("is_stored", False),
                "importance_score": user_memory_trace.get("importance_score", 0.0),
                "tier": user_memory_trace.get("tier"),
                "compression_level": user_memory_trace.get("compression_level", 0),
                "score_breakdown": user_memory_trace.get("score_breakdown", {}),
            },
            "retrieved_memories": [
                {
                    "similarity": round(float(item["similarity"]), 4),
                    "memory": memory_dict(item["memory"]),
                }
                for item in retrieved_raw
            ],
            "assistant_memory": {
                "memory": assistant_mem_data,
                "action": assistant_memory_trace.get("action", "NONE"),
                "decision_reason": assistant_memory_trace.get("decision_reason", ""),
                "merged_content": assistant_memory_trace.get("merged_content"),
                "old_content": assistant_memory_trace.get("old_content"),
                "candidates": assistant_memory_trace.get("candidates", []),
                "is_stored": assistant_memory_trace.get("is_stored", False),
                "importance_score": assistant_memory_trace.get("importance_score", 0.0),
                "tier": assistant_memory_trace.get("tier"),
                "compression_level": assistant_memory_trace.get("compression_level", 0),
                "score_breakdown": assistant_memory_trace.get("score_breakdown", {}),
            },
            "pipeline_stages": pipeline_stages,
            "llm_error": llm_error,
        }

    def transition_memory(self, memory_id: str, target_tier: str) -> Memory:
        """Apply one explicit compressed lifecycle transition."""
        if not self.compression:
            raise RuntimeError("LLM compression is not configured")
        memory = self.storage.get_memory(memory_id)
        if memory is None:
            raise KeyError(memory_id)
        return self.compression.transition(memory, target_tier)