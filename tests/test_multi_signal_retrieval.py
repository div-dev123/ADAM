"""Comprehensive tests for Phase 6: Multi-Signal Retrieval + Context Assembly.

Covers:
1. Multi-signal ranking across 6 signals (semantic, relevance, importance, recency, frequency, tier).
2. Configurable weights and ablation experiments.
3. Context budgeting and token limit enforcement.
4. Semantic redundancy filtering and diagnostic tracking.
5. Dedicated ContextBuilder output and LLM prompt assembly.
6. API endpoint integration (/retrieve and /chat metadata exposure).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.memory.models import Memory, utc_now
from app.memory.storage import SQLiteStorage
from app.memory.tiers import ARCHIVE, LONG_TERM, SHORT_TERM, WORKING
from app.retrieval.context_builder import (
    ContextBudgetConfig,
    ContextBudgeter,
    ContextBuilder,
    estimate_tokens,
)
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.ranking import MultiSignalRanker, RankingWeights
from app.retrieval.retrieval import RetrievalService


# ===========================================================================
# 1. Multi-Signal Ranking Tests
# ===========================================================================

def test_multi_signal_ranking_combines_all_six_signals():
    """Verify that all six signals contribute to the unified final score."""
    ranker = MultiSignalRanker(
        RankingWeights(
            semantic_similarity=0.35,
            query_relevance=0.20,
            importance=0.15,
            recency=0.10,
            access_frequency=0.10,
            tier=0.10,
        )
    )

    now = datetime(2026, 9, 27, 12, 0, 0, tzinfo=timezone.utc)
    mem = Memory(
        memory_id="mem-1",
        user_id="u1",
        content="FastAPI async background tasks using Celery and Redis.",
        embedding=[0.1] * 384,
        created_at=now - timedelta(days=1),
        last_accessed=now - timedelta(hours=2),
        access_count=4,
        importance_score=0.85,
        tier=WORKING,
    )

    scored = ranker.score_candidate(
        memory=mem,
        dense_score=0.90,
        bm25_score=3.5,
        rrf_score=0.03,
        now=now,
    )

    assert "final_score" in scored
    assert "ranking_score" in scored
    assert 0.0 <= scored["final_score"] <= 1.0

    signals = scored["signals"]
    assert "semantic_similarity" in signals
    assert "query_relevance" in signals
    assert "importance" in signals
    assert "recency" in signals
    assert "access_frequency" in signals
    assert "tier" in signals

    # Verify individual signals are normalized [0, 1]
    for name, sig in signals.items():
        assert 0.0 <= sig["value"] <= 1.0
        assert sig["contribution"] >= 0.0

    assert signals["importance"]["value"] == 0.85
    assert signals["tier"]["value"] == 1.0  # WORKING tier score
    assert signals["semantic_similarity"]["value"] == 0.90
    assert scored["selection_reason"] != ""


def test_ranking_weights_ablation_experiments():
    """Perform weight ablation: zeroing out other weights isolates specific signals."""
    now = datetime(2026, 9, 27, 12, 0, 0, tzinfo=timezone.utc)

    # Candidate A: high semantic similarity (0.95), low importance (0.20)
    mem_a = Memory(
        memory_id="A",
        user_id="u1",
        content="Candidate A",
        embedding=[0.1] * 10,
        created_at=now,
        last_accessed=now,
        importance_score=0.20,
        tier=ARCHIVE,
    )

    # Candidate B: low semantic similarity (0.40), high importance (0.95)
    mem_b = Memory(
        memory_id="B",
        user_id="u1",
        content="Candidate B",
        embedding=[0.1] * 10,
        created_at=now,
        last_accessed=now,
        importance_score=0.95,
        tier=WORKING,
    )

    cands = [
        {"memory": mem_a, "dense_score": 0.95, "bm25_score": 0.0},
        {"memory": mem_b, "dense_score": 0.40, "bm25_score": 0.0},
    ]

    # Ablation 1: Semantic-only ranker (all other weights 0.0)
    sem_ranker = MultiSignalRanker(
        RankingWeights(
            semantic_similarity=1.0,
            query_relevance=0.0,
            importance=0.0,
            recency=0.0,
            access_frequency=0.0,
            tier=0.0,
        )
    )
    ranked_sem = sem_ranker.rank_candidates(cands, now=now)
    assert ranked_sem[0]["memory"].memory_id == "A"
    assert ranked_sem[0]["final_score"] == 0.95

    # Ablation 2: Importance-only ranker (all other weights 0.0)
    imp_ranker = MultiSignalRanker(
        RankingWeights(
            semantic_similarity=0.0,
            query_relevance=0.0,
            importance=1.0,
            recency=0.0,
            access_frequency=0.0,
            tier=0.0,
        )
    )
    ranked_imp = imp_ranker.rank_candidates(cands, now=now)
    assert ranked_imp[0]["memory"].memory_id == "B"
    assert ranked_imp[0]["final_score"] == 0.95

    # Ablation 3: Tier-only ranker
    tier_ranker = MultiSignalRanker(
        RankingWeights(
            semantic_similarity=0.0,
            query_relevance=0.0,
            importance=0.0,
            recency=0.0,
            access_frequency=0.0,
            tier=1.0,
        )
    )
    ranked_tier = tier_ranker.rank_candidates(cands, now=now)
    assert ranked_tier[0]["memory"].memory_id == "B"  # WORKING (1.0) beats ARCHIVE (0.4)


# ===========================================================================
# 2. Redundancy Removal Tests
# ===========================================================================

def test_redundancy_removal_filters_similar_memories():
    """Verify that near-duplicate memories are pruned and flagged with an audit reason."""
    embeddings = EmbeddingService("all-MiniLM-L6-v2")
    budgeter = ContextBudgeter(ContextBudgetConfig(redundancy_threshold=0.75, token_budget=1000))

    # Two semantically near-identical memories
    mem1_text = "I love programming in Python for backend development."
    mem2_text = "I enjoy coding in Python for backend engineering builds."
    # One distinct memory
    mem3_text = "The server database is PostgreSQL running on port 5432."

    now = utc_now()
    mem1 = Memory("m1", "u1", mem1_text, embeddings.encode(mem1_text), now, now, importance_score=0.8)
    mem2 = Memory("m2", "u1", mem2_text, embeddings.encode(mem2_text), now, now, importance_score=0.75)
    mem3 = Memory("m3", "u1", mem3_text, embeddings.encode(mem3_text), now, now, importance_score=0.9)

    candidates = [
        {"memory": mem1, "final_score": 0.92, "similarity": 0.92},
        {"memory": mem2, "final_score": 0.88, "similarity": 0.88},
        {"memory": mem3, "final_score": 0.80, "similarity": 0.80},
    ]

    result = budgeter.select_memories(candidates, max_memories=5)

    # mem2 should be pruned as redundant with mem1
    selected_ids = [item["memory"].memory_id for item in result.selected]
    assert "m1" in selected_ids
    assert "m3" in selected_ids
    assert "m2" not in selected_ids

    assert len(result.discarded) >= 1
    discarded_m2 = next(d for d in result.discarded if d["memory"].memory_id == "m2")
    assert discarded_m2["action"] == "PRUNED_REDUNDANT"
    assert "Redundant with selected memory" in discarded_m2["reason"]


# ===========================================================================
# 3. Context Budgeting Tests
# ===========================================================================

def test_context_budget_limits_token_consumption():
    """Verify that selection stops once the token budget is reached."""
    budgeter = ContextBudgeter(ContextBudgetConfig(token_budget=35, redundancy_threshold=0.95))
    now = utc_now()

    mem1 = Memory("m1", "u1", "First short fact about Python framework.", [1.0] + [0.0] * 9, now, now)
    mem2 = Memory("m2", "u1", "Second short fact about PostgreSQL database.", [0.0, 1.0] + [0.0] * 8, now, now)
    mem3 = Memory("m3", "u1", "Third very long fact discussing architecture and microservice container deployment with Kubernetes.", [0.0, 0.0, 1.0] + [0.0] * 7, now, now)

    candidates = [
        {"memory": mem1, "final_score": 0.90},
        {"memory": mem2, "final_score": 0.85},
        {"memory": mem3, "final_score": 0.80},
    ]

    result = budgeter.select_memories(candidates, token_budget=25, max_memories=10)

    # mem1 and mem2 consume approx 8 + 8 = ~16 tokens. mem3 would exceed 25 tokens.
    assert len(result.selected) == 2
    assert result.total_tokens <= 25
    assert len(result.discarded) == 1
    assert result.discarded[0]["action"] == "PRUNED_BUDGET"
    assert "Exceeds context budget" in result.discarded[0]["reason"]


# ===========================================================================
# 4. Context Builder Tests
# ===========================================================================

def test_dedicated_context_builder_formatting():
    """Verify ContextBuilder produces a clean prompt injection block."""
    now = utc_now()
    builder = ContextBuilder()

    mem1 = Memory("m1", "u1", "User prefers dark mode themes.", [0.1], now, now, importance_score=0.85, tier=WORKING)
    mem2 = Memory("m2", "u1", "User works as a Senior Staff Engineer.", [0.2], now, now, importance_score=0.90, tier=LONG_TERM)

    selected = [
        {"memory": mem1, "final_score": 0.91, "similarity": 0.91},
        {"memory": mem2, "final_score": 0.88, "similarity": 0.88},
    ]

    prompt_context = builder.build_context(selected)

    assert "=== RETRIEVED RELEVANT MEMORIES ===" in prompt_context
    assert "[WORKING | Importance: 0.85 | Score: 0.91] User prefers dark mode themes." in prompt_context
    assert "[LONG_TERM | Importance: 0.90 | Score: 0.88] User works as a Senior Staff Engineer." in prompt_context
    assert "===================================" in prompt_context

    # Exclude text matching active message
    prompt_with_exclude = builder.build_context(selected, exclude_text="User prefers dark mode themes.")
    assert "User prefers dark mode themes." not in prompt_with_exclude
    assert "Senior Staff Engineer" in prompt_with_exclude


# ===========================================================================
# 5. Service & API End-to-End Multi-Signal Tests
# ===========================================================================

def test_service_search_with_multi_signal_and_budget(tmp_path: Path):
    """Verify RetrievalService.search integrates ranker, budgeter, and telemetry."""
    storage = SQLiteStorage(tmp_path / "test_ms_retrieval.db")
    embeddings = EmbeddingService("all-MiniLM-L6-v2")

    retrieval = RetrievalService(
        storage=storage,
        embeddings=embeddings,
        ranking_weights=RankingWeights(
            semantic_similarity=0.40,
            query_relevance=0.20,
            importance=0.20,
            tier=0.20,
        ),
        budget_config=ContextBudgetConfig(token_budget=500, redundancy_threshold=0.85),
    )

    user_id = "user-multi"
    retrieval.store_memory(user_id, "FastAPI application with PostgreSQL.")
    retrieval.store_memory(user_id, "FastAPI service configured on port 8000.")
    retrieval.store_memory(user_id, "Redis cache used for session storage.")

    results = retrieval.search(user_id, "FastAPI port", top_k=2)

    assert len(results) <= 2
    for item in results:
        assert "final_score" in item
        assert "ranking_score" in item
        assert "selection_reason" in item
        assert "signals" in item
        assert "signal_scores" in item
        assert "dense_score" in item
        assert "bm25_score" in item

    assert retrieval.last_budget_result is not None
    assert retrieval.last_budget_result.total_tokens > 0


def test_api_retrieve_returns_multi_signal_and_budget_telemetry(tmp_path: Path):
    """Test POST /retrieve API endpoint returns all new Phase 6 metadata."""
    storage = SQLiteStorage(tmp_path / "test_api_ms.db")
    embeddings = EmbeddingService("all-MiniLM-L6-v2")
    user_id = "u-api-test"

    retrieval = RetrievalService(storage=storage, embeddings=embeddings)
    retrieval.store_memory(user_id, "Developing neural network models using PyTorch.")
    retrieval.store_memory(user_id, "Configured Docker containers on AWS ECS.")

    app.state.retrieval = retrieval

    with TestClient(app) as client:
        resp = client.post(
            "/retrieve",
            json={
                "user_id": user_id,
                "query": "PyTorch neural networks",
                "top_k": 2,
                "token_budget": 500,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        assert "results" in data
        assert len(data["results"]) >= 1

        top_hit = data["results"][0]
        assert "final_score" in top_hit
        assert "ranking_score" in top_hit
        assert "selection_reason" in top_hit
        assert "signals" in top_hit
        assert "signal_scores" in top_hit

        # Check context_budget block
        assert "context_budget" in data
        assert data["context_budget"] is not None
        assert "total_tokens" in data["context_budget"]
        assert data["context_budget"]["token_budget"] == 500


def test_api_chat_turn_returns_ranking_and_budget_metadata(tmp_path: Path):
    """Test POST /chat returns pipeline stages and enriched retrieved memories."""
    storage = SQLiteStorage(tmp_path / "test_api_chat.db")
    embeddings = EmbeddingService("all-MiniLM-L6-v2")
    user_id = "u-chat-test"

    retrieval = RetrievalService(storage=storage, embeddings=embeddings)
    retrieval.store_memory(user_id, "My favourite language is Rust for systems programming.")

    app.state.retrieval = retrieval

    with TestClient(app) as client:
        resp = client.post(
            "/chat",
            json={
                "user_id": user_id,
                "message": "Which programming language do I prefer?",
                "top_k": 2,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        # Check pipeline stages
        stages = [s["stage"] for s in data["pipeline_stages"]]
        assert "Multi-Signal Retrieval & Context Budgeting" in stages

        # Check retrieved memories metadata
        retrieved = data["retrieved_memories"]
        assert len(retrieved) >= 1
        item = retrieved[0]
        assert "final_score" in item
        assert "selection_reason" in item
        assert "signals" in item
        assert "signal_scores" in item

        # Check context_budget block
        assert "context_budget" in data
        assert data["context_budget"] is not None
        assert data["context_budget"]["total_tokens"] > 0
