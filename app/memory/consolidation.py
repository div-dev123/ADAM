"""LLM-assisted and heuristic-first consolidation for the Phase 3 write path."""

import re
from dataclasses import dataclass

from app.llm.client import LLMClient
from app.memory.importance import FUNCTION_WORDS, is_filler
from app.memory.models import Memory, utc_now
from app.memory.storage import SQLiteStorage, create_memory
from app.retrieval.similarity import cosine_similarity


@dataclass(frozen=True)
class ConsolidationConfig:
    candidate_limit: int = 3
    min_similarity: float = 0.35
    heuristic_duplicate_threshold: float = 0.92
    heuristic_new_threshold: float = 0.60


def _extract_content_words(text: str) -> set[str]:
    """Extract lowercase non-stop words with length >= 3."""
    words = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    return {w for w in words if w not in FUNCTION_WORDS and len(w) >= 3}


def _verify_merge_quality(old_text: str, new_text: str, merged_text: str) -> bool:
    """Verify that merged text retains core content terms from both parent memories."""
    if not merged_text or not merged_text.strip():
        return False
    old_words = _extract_content_words(old_text)
    new_words = _extract_content_words(new_text)
    merged_words = _extract_content_words(merged_text)

    # If either input had key words, ensure at least some overlap from both
    if old_words and not (old_words & merged_words):
        return False
    if new_words and not (new_words & merged_words):
        return False
    return True


class ConsolidationService:
    def __init__(self, storage: SQLiteStorage, llm: LLMClient, embeddings,
                 scorer, lifecycle_policy, config: ConsolidationConfig):
        self.storage = storage
        self.llm = llm
        self.embeddings = embeddings
        self.scorer = scorer
        self.lifecycle_policy = lifecycle_policy
        self.config = config

    def process(self, user_id: str, content: str, source_role: str = "user") -> Memory | None:
        trace = self.process_with_trace(user_id, content, source_role=source_role)
        return trace.get("memory")

    def process_with_trace(self, user_id: str, content: str, source_role: str = "user") -> dict:
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

        # Calculate score and breakdown for transparent inspection
        score_breakdown = (
            self.scorer.score_with_breakdown(content)
            if hasattr(self.scorer, "score_with_breakdown")
            else {"total": self.scorer.score(content), "signals": {}}
        )

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

        target = candidates[0][1] if candidates else None
        top_similarity = float(candidates[0][0]) if candidates else 0.0
        old_content = target.content if target else None

        # -------------------------------------------------------------------
        # Tiered Decision Strategy: Heuristic-First, LLM-Second
        # -------------------------------------------------------------------
        decision = None

        # 0. Source-Role Isolation: Assistant responses are never merged into user queries
        if source_role == "assistant" and target:
            # If incoming assistant response has candidates, it should not merge into existing user queries
            decision = type("HeuristicDecision", (), {
                "action": "NEW",
                "reason": "Source role isolation (assistant response stored as independent memory)",
                "merged_content": None,
                "merged_text": lambda self: "",
            })()

        # 1. Heuristic DUPLICATE: exact content match with high similarity (>= 0.92)
        elif top_similarity >= self.config.heuristic_duplicate_threshold and target and target.content.strip().lower() == content.strip().lower():
            decision = type("HeuristicDecision", (), {
                "action": "DUPLICATE",
                "reason": f"Heuristic duplicate match (exact match and cosine similarity {top_similarity:.4f} >= {self.config.heuristic_duplicate_threshold})",
                "merged_content": None,
                "merged_text": lambda self: target.content,
            })()

        # 2. Heuristic NEW: no candidates above min_similarity
        elif not candidates:
            decision = type("HeuristicDecision", (), {
                "action": "NEW",
                "reason": "No similar candidates found in memory",
                "merged_content": None,
                "merged_text": lambda self: "",
            })()

        # 3. Middle Ambiguous Band: Delegate to LLM classifier
        else:
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

        # -------------------------------------------------------------------
        # Action Execution & Safety Guards
        # -------------------------------------------------------------------
        if decision.action == "DUPLICATE" and target:
            accessed_at = utc_now()
            self.storage.update_access_metadata(target.memory_id, accessed_at)
            target.last_accessed = accessed_at
            target.updated_at = accessed_at
            target.access_count += 1
            self.storage.record_history(target, "DUPLICATE", target.content, decision.reason)
            memory = target

        elif decision.action == "RELATED" and target:
            # Merge Quality Guard: ensure merged_content is valid and preserves key terms
            merged_candidate = getattr(decision, "merged_content", "") or ""
            if not merged_candidate.strip() or not _verify_merge_quality(target.content, content, merged_candidate):
                decision = type("FallbackDecision", (), {
                    "action": "NEW",
                    "reason": f"Fallback to NEW: RELATED merge failed quality verification or was empty",
                    "merged_content": None,
                    "merged_text": lambda self: "",
                })()
                memory = self._create(user_id, content, embedding)
            else:
                try:
                    memory = self._update(target, content, decision)
                except ValueError:
                    memory = self._create(user_id, content, embedding)
                    decision = type("FallbackDecision", (), {
                        "action": "NEW",
                        "reason": "Fallback to NEW: merged_content was invalid",
                        "merged_content": None,
                        "merged_text": lambda self: "",
                    })()

        elif decision.action == "CONTRADICTORY" and target:
            # Safe CONTRADICTORY:
            # Update the prior memory with superseded_by reference without destroying audit trails
            new_mem = self._create(user_id, content, embedding)
            target.superseded_by = new_mem.memory_id
            target.updated_at = utc_now()
            self.storage.update_memory(target)
            self.storage.record_history(
                target, "CONTRADICTORY", target.content,
                f"Superseded by memory {new_mem.memory_id}: {decision.reason}"
            )
            memory = new_mem

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
            "score_breakdown": score_breakdown.get("signals", {}),
        }

    def _candidates(self, user_id: str, embedding):
        ranked = sorted(
            ((cosine_similarity(embedding, memory.embedding), memory)
             for memory in self.storage.get_memories(user_id)
             if not memory.superseded_by),  # Exclude superseded memories from active consolidation
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


