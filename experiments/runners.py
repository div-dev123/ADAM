"""Evaluation runners for baselines and ADAM ablations."""

import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional
import uuid

from app.memory.consolidation import ConsolidationConfig, ConsolidationService
from app.memory.importance import HeuristicImportanceScorer, ImportanceWeights, is_filler
from app.memory.lifecycle import LifecyclePolicyConfig, MemoryLifecycleManager
from app.memory.models import Memory, utc_now
from app.memory.storage import SQLiteStorage, create_memory
from app.memory.tiers import ARCHIVE, LONG_TERM, SHORT_TERM, WORKING, TierAssigner
from app.retrieval.context_builder import (
    ContextBudgetConfig,
    ContextBudgeter,
    ContextBuilder,
)
from app.retrieval.drift import DriftConfig, QueryDriftDetector
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.hybrid import HybridSearchService
from app.retrieval.ranking import MultiSignalRanker, RankingWeights
from app.retrieval.retrieval import RetrievalService
from app.retrieval.similarity import cosine_similarity
from experiments.configs import ExperimentConfig
from experiments.dataset import ConversationTurn


class BaseRunner:
    """Abstract base class for all experiment runners."""

    def __init__(self, config: ExperimentConfig, embeddings: EmbeddingService, work_dir: Path):
        self.config = config
        self.embeddings = embeddings
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.work_dir / f"eval_{config.name}_{uuid.uuid4().hex[:8]}.db"
        self.storage = SQLiteStorage(self.db_path)

    def ingest(self, user_id: str, turns: List[ConversationTurn]) -> int:
        raise NotImplementedError

    def retrieve(self, user_id: str, query: str) -> List[str]:
        raise NotImplementedError

    def get_stored_memory_count(self, user_id: str) -> int:
        return len([m for m in self.storage.get_memories(user_id) if not m.superseded_by])

    def cleanup(self) -> None:
        try:
            if self.db_path.exists():
                self.db_path.unlink()
        except Exception:
            pass


class RawHistoryRunner(BaseRunner):
    """Baseline 1: Appends all messages chronologically without semantic search or filtering."""

    def __init__(self, config: ExperimentConfig, embeddings: EmbeddingService, work_dir: Path):
        super().__init__(config, embeddings, work_dir)
        self.history: List[str] = []

    def ingest(self, user_id: str, turns: List[ConversationTurn]) -> int:
        for t in turns:
            self.history.append(t.content)
            # Store dummy record for tracking
            dummy_mem = create_memory(user_id, t.content, [0.0])
            self.storage.save_memory(dummy_mem)
        return len(self.history)

    def retrieve(self, user_id: str, query: str) -> List[str]:
        # Raw history baseline: returns most recent turns up to budget/top_k
        k = self.config.top_k
        return list(reversed(self.history[-k:]))


class VectorOnlyRunner(BaseRunner):
    """Baseline 2: Pure dense vector retrieval (top-k cosine similarity), no metadata or drift."""

    def ingest(self, user_id: str, turns: List[ConversationTurn]) -> int:
        for t in turns:
            emb = self.embeddings.encode(t.content)
            mem = create_memory(user_id, t.content, emb)
            mem.importance_score = 0.5
            mem.tier = WORKING
            self.storage.save_memory(mem)
        return len(self.storage.get_memories(user_id))

    def retrieve(self, user_id: str, query: str) -> List[str]:
        query_emb = self.embeddings.encode(query)
        memories = self.storage.get_memories(user_id)
        if not memories:
            return []

        scored = []
        for m in memories:
            score = cosine_similarity(query_emb, m.embedding)
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self.config.top_k]
        return [m.content for _, m in top]


class ImportanceOnlyRunner(BaseRunner):
    """Baseline 3: Ranks candidates solely by intrinsic importance score without multi-signal fusion."""

    def __init__(self, config: ExperimentConfig, embeddings: EmbeddingService, work_dir: Path):
        super().__init__(config, embeddings, work_dir)
        self.scorer = HeuristicImportanceScorer(ImportanceWeights())

    def ingest(self, user_id: str, turns: List[ConversationTurn]) -> int:
        for t in turns:
            emb = self.embeddings.encode(t.content)
            mem = create_memory(user_id, t.content, emb)
            mem.importance_score = self.scorer.score(t.content)
            mem.tier = WORKING
            self.storage.save_memory(mem)
        return len(self.storage.get_memories(user_id))

    def retrieve(self, user_id: str, query: str) -> List[str]:
        memories = self.storage.get_memories(user_id)
        if not memories:
            return []
        # Rank purely by importance
        sorted_m = sorted(memories, key=lambda m: m.importance_score, reverse=True)
        top = sorted_m[: self.config.top_k]
        return [m.content for m in top]


class SlidingWindowRunner(BaseRunner):
    """Baseline 4: LangChain ConversationBufferWindowMemory.
    
    Retains only the most recent K turns in a rolling FIFO queue.
    Facts from turns prior to (N - K) are completely dropped from context.
    """

    def __init__(self, config: ExperimentConfig, embeddings: EmbeddingService, work_dir: Path):
        super().__init__(config, embeddings, work_dir)
        self.window: List[str] = []

    def ingest(self, user_id: str, turns: List[ConversationTurn]) -> int:
        for t in turns:
            self.window.append(t.content)
            dummy_mem = create_memory(user_id, t.content, [0.0])
            self.storage.save_memory(dummy_mem)
        return len(self.window)

    def retrieve(self, user_id: str, query: str) -> List[str]:
        # Return the last K messages (default 6), preserving conversational order
        k = self.config.top_k
        return self.window[-k:]


class StanfordAgentRunner(BaseRunner):
    """Baseline 5: Generative Agents (Park et al. 2023).
    
    Ranks memories by a 3-signal linear formula:
      Score = alpha * Recency + beta * Importance + gamma * Relevance
    No tier hierarchy, no BM25 lexical search, no query drift routing, no MMR deduplication.
    """

    def __init__(self, config: ExperimentConfig, embeddings: EmbeddingService, work_dir: Path):
        super().__init__(config, embeddings, work_dir)
        self.scorer = HeuristicImportanceScorer(ImportanceWeights())
        self.timestamps: Dict[str, float] = {}
        self.decay_factor: float = 0.995

    def ingest(self, user_id: str, turns: List[ConversationTurn]) -> int:
        for idx, t in enumerate(turns):
            emb = self.embeddings.encode(t.content)
            mem = create_memory(user_id, t.content, emb)
            mem.importance_score = self.scorer.score(t.content)
            mem.tier = WORKING
            self.storage.save_memory(mem)
            # Simulated hour-based timestamp index
            self.timestamps[mem.memory_id] = float(idx)
        return len(self.storage.get_memories(user_id))

    def retrieve(self, user_id: str, query: str) -> List[str]:
        query_emb = self.embeddings.encode(query)
        memories = self.storage.get_memories(user_id)
        if not memories:
            return []

        curr_time = float(len(self.timestamps))
        scored = []
        for m in memories:
            # 1. Relevance: Cosine similarity
            relevance = max(0.0, cosine_similarity(query_emb, m.embedding))
            # 2. Importance: Normalized 0 to 1
            importance = float(m.importance_score)
            # 3. Recency: Exponential decay based on game-hour distance
            t_event = self.timestamps.get(m.memory_id, 0.0)
            hours_passed = max(0.0, curr_time - t_event)
            recency = self.decay_factor ** hours_passed

            # Stanford 3-signal linear combination (equal weights alpha=1, beta=1, gamma=1)
            score = relevance + importance + recency
            scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self.config.top_k]
        return [m.content for _, m in top]


class MemGPTRunner(BaseRunner):
    """Baseline 6: MemGPT / Letta OS-Style Memory Agent Architecture.
    
    Models two memory partitions:
      1. Core Memory (in-context persona/human profile block of extracted key facts)
      2. Archival Memory (dense vector store of raw dialogue turns)
    Retrieval injects the active Core Memory block plus top-k Archival search results.
    Does NOT have tier hierarchy (Working/Archive lifecycle), decay pruning, or MMR.
    """

    def __init__(self, config: ExperimentConfig, embeddings: EmbeddingService, work_dir: Path):
        super().__init__(config, embeddings, work_dir)
        self.core_memory_facts: List[str] = []
        self.scorer = HeuristicImportanceScorer(ImportanceWeights())

    def ingest(self, user_id: str, turns: List[ConversationTurn]) -> int:
        for t in turns:
            emb = self.embeddings.encode(t.content)
            mem = create_memory(user_id, t.content, emb)
            score = self.scorer.score(t.content)
            mem.importance_score = score
            mem.tier = WORKING
            self.storage.save_memory(mem)

            # Core memory ingestion: If statement conveys high importance personal traits,
            # simulate MemGPT core_memory_append() into persona block (capped at 5 key facts)
            if score >= 0.70 and len(self.core_memory_facts) < 5:
                # Extract clean core statement
                cleaned = t.content.split("]:")[-1].strip() if "]:" in t.content else t.content
                if cleaned not in self.core_memory_facts:
                    self.core_memory_facts.append(cleaned)

        return len(self.storage.get_memories(user_id))

    def retrieve(self, user_id: str, query: str) -> List[str]:
        # Step 1: Archival Memory search (top-k dense vector similarity)
        query_emb = self.embeddings.encode(query)
        memories = self.storage.get_memories(user_id)
        archival_matches: List[str] = []
        if memories:
            scored = []
            for m in memories:
                sim = cosine_similarity(query_emb, m.embedding)
                scored.append((sim, m))
            scored.sort(key=lambda x: x[0], reverse=True)
            archival_matches = [m.content for _, m in scored[: self.config.top_k]]

        # Step 2: Assemble MemGPT context (Core Memory block + Archival search)
        context_items: List[str] = []
        if self.core_memory_facts:
            core_block = "[Core Memory - User Profile]: " + " | ".join(self.core_memory_facts)
            context_items.append(core_block)
        context_items.extend(archival_matches)

        return context_items[: self.config.top_k]


class ConstantScorer:
    """Mock scorer that returns a constant score for ablation experiments."""
    def __init__(self, constant: float = 0.5):
        self.constant = constant

    def score(self, *args, **kwargs) -> float:
        return self.constant

    def score_with_breakdown(self, text: str) -> dict:
        return {"total": self.constant, "signals": {"constant": self.constant}}


class FlatTierAssigner:
    """Mock tier assigner that puts all memories into a single uniform tier."""
    def initial_tier(self, score: float) -> str:
        return SHORT_TERM


class MockLLMForEval:
    """Mock LLM providing accurate semantic consolidation decisions for offline evaluation experiments."""

    def classify_memory(self, content: str, candidates: list[dict]):
        if not candidates:
            return type("Decision", (), {
                "action": "NEW",
                "reason": "No candidates",
                "merged_content": None,
                "merged_text": lambda self: "",
            })()

        top_cand = candidates[0]
        sim = float(top_cand.get("similarity", 0.0))
        cand_text = top_cand.get("content", "")

        words_new = set(re.findall(r"\w+", content.lower()))
        words_cand = set(re.findall(r"\w+", cand_text.lower()))
        overlap = len(words_new & words_cand) / max(len(words_new | words_cand), 1)

        # Repetition / Reinforcement
        if sim >= 0.55 and overlap >= 0.20:
            richer = content if len(content) > len(cand_text) else cand_text
            return type("Decision", (), {
                "action": "DUPLICATE",
                "reason": f"Semantic repetition consolidation (sim={sim:.3f}, overlap={overlap:.0%})",
                "merged_content": richer,
                "merged_text": lambda self: richer,
            })()

        return type("Decision", (), {
            "action": "NEW",
            "reason": "Distinct semantic information",
            "merged_content": None,
            "merged_text": lambda self: "",
        })()

    def generate_chat_response(self, *args, **kwargs):
        return "Acknowledged."


class AdamRunner(BaseRunner):
    """Full ADAM system runner with complete support for all ablation switches."""

    def __init__(self, config: ExperimentConfig, embeddings: EmbeddingService, work_dir: Path):
        super().__init__(config, embeddings, work_dir)
        self.llm = MockLLMForEval() if config.enable_consolidation else None

        # 1. Importance Scorer
        if config.enable_importance_scoring:
            self.scorer = HeuristicImportanceScorer(ImportanceWeights())
        else:
            self.scorer = ConstantScorer(0.5)

        # 2. Tier Assigner
        if config.enable_tier_hierarchy:
            self.tier_assigner = TierAssigner()
        else:
            self.tier_assigner = FlatTierAssigner()

        # 3. Ranking Weights
        if config.enable_multi_signal:
            r_weights = RankingWeights(
                semantic_similarity=0.35,
                query_relevance=0.20,
                importance=0.15 if config.enable_importance_scoring else 0.0,
                recency=0.10 if config.enable_forgetting else 0.0,
                access_frequency=0.10,
                tier=0.10 if config.enable_tier_hierarchy else 0.0,
            )
        else:
            # Multi-signal disabled: 100% semantic similarity
            r_weights = RankingWeights(
                semantic_similarity=1.0,
                query_relevance=0.0,
                importance=0.0,
                recency=0.0,
                access_frequency=0.0,
                tier=0.0,
            )
        self.ranker = MultiSignalRanker(weights=r_weights)

        # 4. Context Budgeter
        budget_conf = ContextBudgetConfig(
            token_budget=config.token_budget if config.enable_context_budgeting else 999999,
            redundancy_threshold=config.redundancy_threshold if config.enable_redundancy_removal else 1.0,
            max_memories=config.top_k,
        )
        self.budgeter = ContextBudgeter(config=budget_conf)
        self.context_builder = ContextBuilder(budgeter=self.budgeter)

        # 5. Lifecycle Manager
        lifecycle_conf = LifecyclePolicyConfig(
            recency_decay_lambda=0.05 if config.enable_forgetting else 0.0,
            weight_importance=0.40 if config.enable_importance_scoring else 0.0,
        )
        self.lifecycle_manager = MemoryLifecycleManager(
            storage=self.storage,
            config=lifecycle_conf,
        )

        # 6. Drift Detector
        self.drift_detector = QueryDriftDetector(
            embeddings=embeddings,
            config=DriftConfig(),
        )

        # 7. Retrieval Service
        self.service = RetrievalService(
            storage=self.storage,
            embeddings=embeddings,
            llm=self.llm,
            scorer=self.scorer,
            tier_assigner=self.tier_assigner,
            lifecycle_manager=self.lifecycle_manager,
            drift_detector=self.drift_detector,
            ranking_weights=r_weights,
            ranker=self.ranker,
            budget_config=budget_conf,
            budgeter=self.budgeter,
            context_builder=self.context_builder,
            search_mode="hybrid" if config.enable_hybrid_bm25 else "dense",
        )

        # Handle consolidation ablation
        if not config.enable_consolidation:
            self.service.consolidation = None

    def ingest(self, user_id: str, turns: List[ConversationTurn]) -> int:
        for t in turns:
            if not self.config.enable_consolidation:
                # Direct storage without consolidation
                is_fill, _ = is_filler(t.content)
                if not is_fill:
                    emb = self.embeddings.encode(t.content)
                    mem = create_memory(user_id, t.content, emb)
                    mem.importance_score = self.scorer.score(t.content)
                    mem.tier = self.tier_assigner.initial_tier(mem.importance_score)
                    self.storage.save_memory(mem)
            else:
                self.service.store_memory_with_trace(
                    user_id=user_id, content=t.content, source_role=t.role
                )
        return self.get_stored_memory_count(user_id)

    def retrieve(self, user_id: str, query: str) -> List[str]:
        scope_tiers = None
        if not self.config.enable_query_drift or not self.config.enable_tier_hierarchy:
            # Query all tiers flatly
            scope_tiers = [WORKING, SHORT_TERM, LONG_TERM, ARCHIVE]

        results = self.service.search(
            user_id=user_id,
            query=query,
            top_k=self.config.top_k,
            scope_tiers=scope_tiers,
            token_budget=self.config.token_budget if self.config.enable_context_budgeting else 999999,
            redundancy_threshold=self.config.redundancy_threshold if self.config.enable_redundancy_removal else 1.0,
        )
        return [item["memory"].content for item in results]


def create_runner(config: ExperimentConfig, embeddings: EmbeddingService, work_dir: Path) -> BaseRunner:
    """Factory creating the appropriate runner for a given configuration."""
    if config.name == "baseline_raw_history":
        return RawHistoryRunner(config, embeddings, work_dir)
    elif config.name == "baseline_vector_only":
        return VectorOnlyRunner(config, embeddings, work_dir)
    elif config.name == "baseline_importance_only":
        return ImportanceOnlyRunner(config, embeddings, work_dir)
    elif config.name == "baseline_sliding_window":
        return SlidingWindowRunner(config, embeddings, work_dir)
    elif config.name == "baseline_stanford_agent":
        return StanfordAgentRunner(config, embeddings, work_dir)
    elif config.name == "baseline_memgpt":
        return MemGPTRunner(config, embeddings, work_dir)
    else:
        return AdamRunner(config, embeddings, work_dir)
