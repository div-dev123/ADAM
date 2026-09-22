from typing import Optional
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


# Negation / update phrases that strongly signal a CONTRADICTORY update
_CONTRADICTION_TRIGGERS = re.compile(
    r"\b("
    r"instead of|no longer|not anymore|switched (to|from)|changed (to|from)"
    r"|quit|stopped using|gave up|moved (to|away from|on from)"
    r"|replaced .{1,30} with|dropped|ditched|abandoned"
    r"|now use|now prefer|now (hate|dislike|love|like)"
    r"|i (hate|dislike|don'?t like|no longer like|no longer use)"
    r")",
    re.IGNORECASE,
)


def _heuristic_contradiction(
    new_content: str, target_content: str, top_similarity: float
) -> bool:
    """Return True if the new content looks like an explicit update/negation of target.

    Requires:
    1. A strong negation/update trigger phrase in the new content.
    2. At least one shared meaningful content word (topic overlap) between the two.
    3. Similarity >= 0.35 (already guaranteed by the candidate filter).
    """
    if not re.search(_CONTRADICTION_TRIGGERS, new_content):
        return False

    # Shared meaningful words (topic alignment)
    new_words = _extract_content_words(new_content)
    old_words = _extract_content_words(target_content)
    shared = new_words & old_words

    # At least one meaningful word must be shared (e.g. "python", "rust", "java")
    return len(shared) >= 1


class ConsolidationService:
    def __init__(self, storage: SQLiteStorage, llm: LLMClient, embeddings,
                 scorer, lifecycle_policy, config: ConsolidationConfig):
        self.storage = storage
        self.llm = llm
        self.embeddings = embeddings
        self.scorer = scorer
        self.lifecycle_policy = lifecycle_policy
        self.config = config

    def process(self, user_id: str, content: str, source_role: str = "user") -> Optional[Memory]:
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
        # Candidates filtered by same source_role so user facts only compare against user facts
        candidates = self._candidates(user_id, embedding, source_role=source_role)
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

        # 1. Heuristic DUPLICATE: very high semantic similarity (>= 0.95) with substantial word overlap
        elif top_similarity >= 0.95 and target:
            old_words = set(re.findall(r"\w+", target.content.lower()))
            new_words = set(re.findall(r"\w+", content.lower()))
            overlap = len(old_words & new_words) / max(len(old_words | new_words), 1)

            # If vocabulary overlap is >= 85%, this is fundamentally the same statement/query
            if overlap >= 0.85:
                # If the new statement adds minor details (e.g. "which will help to save cost"), prefer the richer version
                merged = content if len(content) > len(target.content) else target.content
                decision = type("HeuristicDecision", (), {
                    "action": "DUPLICATE",
                    "reason": f"Heuristic duplicate match (similarity {top_similarity:.4f} >= 0.95, {overlap:.0%} vocabulary overlap)",
                    "merged_content": merged,
                    "merged_text": lambda self: merged,
                })()
            else:
                # High similarity but different content -> Send to LLM with candidate context
                try:
                    decision = self.llm.classify_memory(content, [
                        {"memory_id": c["memory_id"], "content": c["content"], "similarity": c["similarity"]}
                        for c in candidates_info
                    ])
                except Exception as error:
                    decision = type("FallbackDecision", (), {
                        "action": "NEW",
                        "reason": f"Fallback to NEW (LLM unavailable: {error})",
                        "merged_content": None,
                        "merged_text": lambda self: "",
                    })()

        # 2. Heuristic CONTRADICTORY: scan ALL candidates (not just top-1) for an explicit negation
        elif target:
            contradictory_target = None
            for sim, cand in candidates:
                if _heuristic_contradiction(content, cand.content, float(sim)):
                    contradictory_target = (sim, cand)
                    break

            if contradictory_target:
                contra_sim, contra_cand = contradictory_target
                merged = content  # The new statement IS the current truth
                decision = type("HeuristicDecision", (), {
                    "action": "CONTRADICTORY",
                    "reason": f"Heuristic contradiction: explicit update/negation detected (shared topic with candidate sim={float(contra_sim):.3f})",
                    "merged_content": merged,
                    "merged_text": lambda self: merged,
                    "_contra_target": contra_cand,
                })()
                # Override target to be the memory we are actually superseding
                target = contra_cand
            else:
                # No contradiction found — delegate to LLM for RELATED vs NEW
                try:
                    decision = self.llm.classify_memory(content, [
                        {"memory_id": c["memory_id"], "content": c["content"], "similarity": c["similarity"]}
                        for c in candidates_info
                    ])
                except Exception as error:
                    decision = type("FallbackDecision", (), {
                        "action": "NEW",
                        "reason": f"Fallback to NEW (LLM unavailable: {error})",
                        "merged_content": None,
                        "merged_text": lambda self: "",
                    })()

        # 3. Heuristic NEW: no candidates above min_similarity
        elif not candidates:
            decision = type("HeuristicDecision", (), {
                "action": "NEW",
                "reason": "No similar candidates found in memory",
                "merged_content": None,
                "merged_text": lambda self: "",
            })()


        # -------------------------------------------------------------------
        # Action Execution & Safety Guards
        # -------------------------------------------------------------------
        if decision.action == "DUPLICATE" and target:
            accessed_at = utc_now()
            prior_content = target.content
            # If the duplicate statement has an updated/richer representation, adopt it
            if decision.merged_content and decision.merged_content != target.content:
                target.content = decision.merged_content
                target.embedding = self.embeddings.encode(target.content)
                target.importance_score = self.scorer.score(
                    target.content, target.access_count, target.created_at
                )
                self.storage.update_memory(target)

            self.storage.update_access_metadata(target.memory_id, accessed_at)
            target.last_accessed = accessed_at
            target.updated_at = accessed_at
            target.access_count += 1
            self.storage.record_history(target, "DUPLICATE", prior_content, decision.reason)
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
                memory = self._create(user_id, content, embedding, source_role=source_role)
            else:
                try:
                    memory = self._update(target, content, decision)
                except ValueError:
                    memory = self._create(user_id, content, embedding, source_role=source_role)
                    decision = type("FallbackDecision", (), {
                        "action": "NEW",
                        "reason": "Fallback to NEW: merged_content was invalid",
                        "merged_content": None,
                        "merged_text": lambda self: "",
                    })()

        elif decision.action == "CONTRADICTORY" and target:
            # Safe CONTRADICTORY:
            # The prior memory is archived as superseded, and the new memory takes its active operational tier
            prior_tier = target.tier
            new_mem = self._create(user_id, content, embedding, source_role=source_role)
            if prior_tier == "WORKING" and new_mem.tier != "WORKING":
                new_mem.tier = "WORKING"
                new_mem.importance_score = max(new_mem.importance_score, target.importance_score)
                self.storage.update_memory(new_mem)

            target.superseded_by = new_mem.memory_id
            target.tier = "ARCHIVE"
            target.compression_level = max(getattr(target, "compression_level", 0), 2)
            target.updated_at = utc_now()
            self.storage.update_memory(target)
            self.storage.record_history(
                target, "CONTRADICTORY", target.content,
                f"Superseded by memory {new_mem.memory_id}: {decision.reason}",
                new_content=new_mem.content,
            )
            self.storage.record_history(
                new_mem, "CONTRADICTORY", target.content,
                f"Supersedes prior memory {target.memory_id}: {decision.reason}",
                new_content=new_mem.content,
            )
            memory = new_mem

        else:
            memory = self._create(user_id, content, embedding, source_role=source_role)

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

    def _candidates(self, user_id: str, embedding, source_role: str = "user"):
        ranked = sorted(
            ((cosine_similarity(embedding, memory.embedding), memory)
             for memory in self.storage.get_memories(user_id)
             if not memory.superseded_by
             and getattr(memory, "source_role", "user") == source_role),
            key=lambda item: item[0], reverse=True,
        )
        return [item for item in ranked if item[0] >= self.config.min_similarity][
            : self.config.candidate_limit
        ]

    def _create(self, user_id, content, embedding, source_role: str = "user"):
        memory = create_memory(user_id, content, embedding, source_role=source_role)
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


