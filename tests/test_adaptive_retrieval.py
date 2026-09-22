"""Unit and integration tests for Phase 5: Adaptive Retrieval + Query Drift."""

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.memory.models import Memory, utc_now
from app.memory.storage import SQLiteStorage
from app.memory.tiers import ARCHIVE, LONG_TERM, SHORT_TERM, WORKING
from app.retrieval.drift import DriftConfig, QueryDriftDetector
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retrieval import RetrievalService


def create_test_memories(storage: SQLiteStorage, user_id: str, embeddings: EmbeddingService):
    now = utc_now()
    memories = [
        # WORKING tier memory
        Memory(
            memory_id="mem-working-1",
            user_id=user_id,
            content="I am developing an AI agent using FastAPI and Python.",
            embedding=embeddings.encode("I am developing an AI agent using FastAPI and Python."),
            created_at=now,
            last_accessed=now,
            access_count=1,
            importance_score=0.85,
            tier=WORKING,
        ),
        # SHORT_TERM tier memory
        Memory(
            memory_id="mem-short-1",
            user_id=user_id,
            content="We discussed using Pydantic models for request validation today.",
            embedding=embeddings.encode("We discussed using Pydantic models for request validation today."),
            created_at=now - timedelta(days=2),
            last_accessed=now - timedelta(days=2),
            access_count=1,
            importance_score=0.55,
            tier=SHORT_TERM,
        ),
        # LONG_TERM tier memory
        Memory(
            memory_id="mem-long-1",
            user_id=user_id,
            content="Architecture decision: Deploy Docker containers to AWS ECS with PostgreSQL.",
            embedding=embeddings.encode("Architecture decision: Deploy Docker containers to AWS ECS with PostgreSQL."),
            created_at=now - timedelta(days=20),
            last_accessed=now - timedelta(days=15),
            access_count=3,
            importance_score=0.78,
            tier=LONG_TERM,
        ),
        # ARCHIVE tier memory
        Memory(
            memory_id="mem-archive-1",
            user_id=user_id,
            content="Old notes: The favorite pizza place is Luigi's Pizzeria downtown.",
            embedding=embeddings.encode("Old notes: The favorite pizza place is Luigi's Pizzeria downtown."),
            created_at=now - timedelta(days=80),
            last_accessed=now - timedelta(days=75),
            access_count=0,
            importance_score=0.20,
            tier=ARCHIVE,
        ),
    ]
    for mem in memories:
        storage.save_memory(mem)
    return memories


def test_low_drift_query_selects_working_and_short_term(tmp_path: Path):
    """Low drift query on continuous topic scopes strictly to WORKING and SHORT_TERM tiers."""
    storage = SQLiteStorage(tmp_path / "test_low_drift.db")
    embeddings = EmbeddingService("all-MiniLM-L6-v2")
    service = RetrievalService(storage=storage, embeddings=embeddings)
    user_id = "user-drift-1"
    create_test_memories(storage, user_id, embeddings)

    # Context is about Python FastAPI development
    context = "I am writing Python code for our FastAPI web API."
    query = "Which Python web framework am I using for the AI agent?"

    results = service.search(user_id=user_id, query=query, top_k=5, context=context)

    drift = service.last_drift_result
    assert drift is not None
    assert drift.drift_level == "LOW"
    assert drift.drift_score < 0.55
    assert drift.scope_tiers == [WORKING, SHORT_TERM]
    assert "Low query drift" in drift.reason

    # Ensure results only come from WORKING or SHORT_TERM
    assert len(results) > 0
    for item in results:
        assert item["memory_tier"] in [WORKING, SHORT_TERM]
        assert item["drift_level"] == "LOW"
        assert item["scope_selection_reason"] == drift.reason
        assert item["memory"].memory_id != "mem-archive-1"


def test_medium_drift_query_includes_long_term(tmp_path: Path):
    """Medium drift query with topic shift expands scope to include LONG_TERM tier."""
    storage = SQLiteStorage(tmp_path / "test_med_drift.db")
    embeddings = EmbeddingService("all-MiniLM-L6-v2")
    service = RetrievalService(storage=storage, embeddings=embeddings)
    user_id = "user-drift-2"
    create_test_memories(storage, user_id, embeddings)

    # Context is about software engineering/backend; Query is about cloud deployment and database (topic evolution)
    context = "We are working on software engineering and Python backend."
    query = "Where are we deploying Docker containers and PostgreSQL?"

    results = service.search(user_id=user_id, query=query, top_k=5, context=context)

    drift = service.last_drift_result
    assert drift is not None
    assert drift.drift_level == "MEDIUM"
    assert 0.55 <= drift.drift_score < 0.85
    assert drift.scope_tiers == [WORKING, SHORT_TERM, LONG_TERM]
    assert "Medium query drift" in drift.reason

    # Verify LONG_TERM memory is reachable
    retrieved_ids = [item["memory"].memory_id for item in results]
    assert "mem-long-1" in retrieved_ids
    # ARCHIVE memory Luigi's pizza is still not in scope
    assert "mem-archive-1" not in retrieved_ids


def test_high_drift_query_searches_all_tiers_including_archive(tmp_path: Path):
    """High drift query radically changing subject expands scope across all tiers, including ARCHIVE."""
    storage = SQLiteStorage(tmp_path / "test_high_drift.db")
    embeddings = EmbeddingService("all-MiniLM-L6-v2")
    service = RetrievalService(storage=storage, embeddings=embeddings)
    user_id = "user-drift-3"
    create_test_memories(storage, user_id, embeddings)

    # Context is software engineering; Query is completely off-topic: food and restaurants
    context = "Refactoring Python async controllers and SQL indices."
    query = "What was the name of Luigi's pizza restaurant downtown?"

    results = service.search(user_id=user_id, query=query, top_k=5, context=context)

    drift = service.last_drift_result
    assert drift is not None
    assert drift.drift_level == "HIGH"
    assert drift.drift_score >= 0.85
    assert drift.scope_tiers == [WORKING, SHORT_TERM, LONG_TERM, ARCHIVE]
    assert "High query drift" in drift.reason

    # Verify ARCHIVE memory Luigi's pizza is retrieved!
    retrieved_ids = [item["memory"].memory_id for item in results]
    assert "mem-archive-1" in retrieved_ids
    assert results[0]["memory_tier"] == ARCHIVE


def test_retrieval_still_working_when_no_previous_context_exists(tmp_path: Path):
    """When no context or chat history is provided, retrieval works seamlessly with default scope."""
    storage = SQLiteStorage(tmp_path / "test_no_context.db")
    embeddings = EmbeddingService("all-MiniLM-L6-v2")
    service = RetrievalService(storage=storage, embeddings=embeddings)
    user_id = "user-drift-4"
    create_test_memories(storage, user_id, embeddings)

    # Calling search with context=None and chat_history=None
    results = service.search(
        user_id=user_id,
        query="FastAPI and Python AI agent",
        top_k=2,
        context=None,
        chat_history=None,
    )

    drift = service.last_drift_result
    assert drift is not None
    assert drift.drift_level == "LOW"
    assert drift.drift_score == 0.0
    assert "No previous conversation context" in drift.reason
    assert len(results) > 0
    assert results[0]["memory"].memory_id == "mem-working-1"


def test_correct_tier_selection_and_metadata(tmp_path: Path):
    """Retrieval metadata includes similarity, memory_tier, importance, drift_level, and scope reason."""
    storage = SQLiteStorage(tmp_path / "test_meta.db")
    embeddings = EmbeddingService("all-MiniLM-L6-v2")
    service = RetrievalService(storage=storage, embeddings=embeddings)
    user_id = "user-drift-5"
    create_test_memories(storage, user_id, embeddings)

    context = "FastAPI backend services."
    query = "FastAPI agent development"

    results = service.search(user_id=user_id, query=query, top_k=1, context=context)
    assert len(results) == 1
    top_hit = results[0]

    assert "similarity" in top_hit
    assert "memory_tier" in top_hit
    assert top_hit["memory_tier"] == WORKING
    assert "importance" in top_hit
    assert top_hit["importance"] == 0.85
    assert "drift_level" in top_hit
    assert top_hit["drift_level"] == "LOW"
    assert "scope_selection_reason" in top_hit
    assert "Low query drift" in top_hit["scope_selection_reason"]


def test_access_count_and_timestamp_updated_on_adaptive_retrieval(tmp_path: Path):
    """Retrieved memories have their access_count incremented and last_accessed updated."""
    storage = SQLiteStorage(tmp_path / "test_access.db")
    embeddings = EmbeddingService("all-MiniLM-L6-v2")
    service = RetrievalService(storage=storage, embeddings=embeddings)
    user_id = "user-drift-6"
    create_test_memories(storage, user_id, embeddings)

    initial_mem = storage.get_memory("mem-working-1")
    assert initial_mem.access_count == 1
    initial_last_accessed = initial_mem.last_accessed

    service.search(user_id=user_id, query="FastAPI development", top_k=1)

    updated_mem = storage.get_memory("mem-working-1")
    assert updated_mem.access_count == 2
    assert updated_mem.last_accessed >= initial_last_accessed


def test_api_retrieve_with_context_and_drift_telemetry(tmp_path: Path):
    """Test POST /retrieve API endpoint with context and drift telemetry."""
    storage = SQLiteStorage(tmp_path / "test_api_drift.db")
    embeddings = EmbeddingService("all-MiniLM-L6-v2")
    user_id = "user-api-drift"
    create_test_memories(storage, user_id, embeddings)

    app.state.retrieval = RetrievalService(storage=storage, embeddings=embeddings)

    with TestClient(app) as client:
        # Request with high-drift query relative to context
        resp = client.post(
            "/retrieve",
            json={
                "user_id": user_id,
                "query": "Where is Luigi's pizza downtown located?",
                "top_k": 2,
                "context": "Writing Python FastAPI endpoints and database transactions.",
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        # Check top-level drift block
        assert "drift" in data
        assert data["drift"]["level"] == "HIGH"
        assert data["drift"]["score"] >= 0.85
        assert ARCHIVE in data["drift"]["scope_tiers"]
        assert "High query drift" in data["drift"]["reason"]

        # Check result items
        assert len(data["results"]) >= 1
        top_item = data["results"][0]
        assert top_item["memory_tier"] == ARCHIVE
        assert top_item["drift_level"] == "HIGH"
        assert "scope_selection_reason" in top_item
        assert "similarity" in top_item


def test_configurable_drift_thresholds(tmp_path: Path):
    """Custom DriftConfig changes the drift level and scope selection for the same query/context."""
    storage = SQLiteStorage(tmp_path / "test_config_drift.db")
    embeddings = EmbeddingService("all-MiniLM-L6-v2")
    user_id = "user-drift-custom"
    create_test_memories(storage, user_id, embeddings)

    # With default config (low: 0.55, high: 0.85), this pair has score ~0.76 which is MEDIUM
    context = "We are working on software engineering and Python backend."
    query = "Where are we deploying Docker containers and PostgreSQL?"

    service_default = RetrievalService(storage=storage, embeddings=embeddings)
    service_default.search(user_id=user_id, query=query, top_k=5, context=context)
    assert service_default.last_drift_result.drift_level == "MEDIUM"

    # With strict high threshold (high_drift_threshold=0.70), the same query triggers HIGH drift
    strict_config = DriftConfig(low_drift_threshold=0.30, high_drift_threshold=0.70)
    detector_strict = QueryDriftDetector(embeddings=embeddings, config=strict_config)
    service_strict = RetrievalService(
        storage=storage, embeddings=embeddings, drift_detector=detector_strict
    )
    service_strict.search(user_id=user_id, query=query, top_k=5, context=context)
    assert service_strict.last_drift_result.drift_level == "HIGH"
    assert service_strict.last_drift_result.scope_tiers == [
        WORKING,
        SHORT_TERM,
        LONG_TERM,
        ARCHIVE,
    ]


