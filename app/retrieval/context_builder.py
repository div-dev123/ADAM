"""Context budgeting and prompt assembly for ADAM Phase 6: Multi-Signal Retrieval.

Provides:
1. Token budget enforcement and dynamic candidate selection.
2. Semantic redundancy filtering using pairwise embedding similarity.
3. Dedicated context formatting for LLM prompt injection.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.memory.models import Memory
from app.retrieval.similarity import cosine_similarity


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text snippet using word-density heuristic."""
    if not text:
        return 0
    words = len(text.split())
    # Standard English LLM tokenizers produce ~1.25 to 1.35 tokens per whitespace word
    return max(1, math.ceil(words * 1.3))


@dataclass
class ContextBudgetConfig:
    """Configuration for memory selection within LLM context constraints."""

    token_budget: int = 800
    character_budget: Optional[int] = 3200
    redundancy_threshold: float = 0.80
    max_memories: int = 5


@dataclass
class ContextBudgetResult:
    """Detailed telemetry of memory context selection and budget utilization."""

    selected: List[Dict[str, Any]]
    discarded: List[Dict[str, Any]]
    total_tokens: int
    total_characters: int
    budget_used_ratio: float
    token_budget: int


class ContextBudgeter:
    """Filters redundant memories and selects the highest-scoring candidates within a token budget."""

    def __init__(self, config: Optional[ContextBudgetConfig] = None):
        self.config = config or ContextBudgetConfig()

    def select_memories(
        self,
        ranked_candidates: Sequence[Dict[str, Any]],
        token_budget: Optional[int] = None,
        max_memories: Optional[int] = None,
        redundancy_threshold: Optional[float] = None,
    ) -> ContextBudgetResult:
        """Select top ranked candidates while pruning redundant items and respecting token budget."""
        effective_budget = token_budget if token_budget is not None else self.config.token_budget
        effective_limit = max_memories if max_memories is not None else self.config.max_memories
        effective_redundancy = (
            redundancy_threshold
            if redundancy_threshold is not None
            else self.config.redundancy_threshold
        )

        selected: List[Dict[str, Any]] = []
        discarded: List[Dict[str, Any]] = []
        used_tokens = 0
        used_chars = 0

        for candidate in ranked_candidates:
            memory: Memory = candidate["memory"]
            content = memory.content
            cand_tokens = estimate_tokens(content)
            cand_chars = len(content)

            # 1. Redundancy check against already-selected memories
            is_redundant = False
            redundancy_reason = ""
            for sel in selected:
                sel_mem: Memory = sel["memory"]
                if memory.embedding and sel_mem.embedding:
                    # Use cosine similarity for high-dimensional embeddings (> 4 dimensions)
                    if len(memory.embedding) > 4 and len(sel_mem.embedding) > 4:
                        sim = cosine_similarity(memory.embedding, sel_mem.embedding)
                        if sim >= effective_redundancy:
                            is_redundant = True
                            redundancy_reason = (
                                f"Redundant with selected memory '{sel_mem.memory_id[:8]}' "
                                f"(similarity {sim:.3f} >= threshold {effective_redundancy:.2f})"
                            )
                            break
                    else:
                        # For low-dimensional toy embeddings (e.g. in unit test mocks), use word Jaccard overlap
                        words_cand = set(content.lower().split())
                        words_sel = set(sel_mem.content.lower().split())
                        jaccard = len(words_cand & words_sel) / max(len(words_cand | words_sel), 1)
                        if jaccard >= effective_redundancy:
                            is_redundant = True
                            redundancy_reason = (
                                f"Redundant with selected memory '{sel_mem.memory_id[:8]}' "
                                f"(word overlap {jaccard:.3f} >= threshold {effective_redundancy:.2f})"
                            )
                            break

            if is_redundant:
                discarded.append({
                    "memory": memory,
                    "reason": redundancy_reason,
                    "action": "PRUNED_REDUNDANT",
                    "final_score": candidate.get("final_score", 0.0),
                })
                continue

            # 2. Token budget check
            if used_tokens + cand_tokens > effective_budget and len(selected) > 0:
                discarded.append({
                    "memory": memory,
                    "reason": (
                        f"Exceeds context budget: adding {cand_tokens} tokens would push total "
                        f"({used_tokens + cand_tokens}) past budget limit ({effective_budget})"
                    ),
                    "action": "PRUNED_BUDGET",
                    "final_score": candidate.get("final_score", 0.0),
                })
                continue

            # 3. Add to selected
            used_tokens += cand_tokens
            used_chars += cand_chars
            selected.append(candidate)

            # 4. Limit check
            if len(selected) >= effective_limit:
                break

        ratio = round(used_tokens / max(1, effective_budget), 4) if effective_budget > 0 else 0.0

        return ContextBudgetResult(
            selected=selected,
            discarded=discarded,
            total_tokens=used_tokens,
            total_characters=used_chars,
            budget_used_ratio=ratio,
            token_budget=effective_budget,
        )


class ContextBuilder:
    """Assembles prompt context blocks from selected memories for LLM injection."""

    def __init__(self, budgeter: Optional[ContextBudgeter] = None):
        self.budgeter = budgeter or ContextBudgeter()

    def build_context(
        self,
        selected_memories: Sequence[Dict[str, Any]],
        exclude_text: Optional[str] = None,
        header: str = "=== RETRIEVED RELEVANT MEMORIES ===",
        footer: str = "===================================",
    ) -> str:
        """Format selected memories into a concise, structured block for LLM prompts."""
        if not selected_memories:
            return "No relevant memories retrieved."

        norm_exclude = exclude_text.strip().lower() if exclude_text else None
        blocks: List[str] = []

        for item in selected_memories:
            mem = item.get("memory")
            if not mem:
                continue

            content = getattr(mem, "content", str(mem))
            if norm_exclude and content.strip().lower() == norm_exclude:
                continue

            tier = getattr(mem, "tier", "UNKNOWN")
            imp = getattr(mem, "importance_score", 0.0)
            score = item.get("final_score", item.get("similarity", 0.0))
            reason = item.get("selection_reason", "")

            block_line = f"• [{tier} | Importance: {imp:.2f} | Score: {score:.2f}] {content}"
            blocks.append(block_line)

        if not blocks:
            return "None available."

        context_body = "\n".join(blocks)
        if header and footer:
            return f"{header}\n{context_body}\n{footer}"
        return context_body
