"""Unit and integration tests for Phase 4: Selective Forgetting and Automatic Memory Lifecycle."""

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app, build_retrieval_service
from app.memory.lifecycle import (
    LifecyclePolicyConfig,
    MemoryLifecycleManager,
)
from app.memory.models import Memory, utc_now
from app.memory.storage import SQLiteStorage, create_memory
from app.memory.tiers import ARCHIVE, LONG_TERM, SHORT_TERM, WORKING
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retrieval import RetrievalService


def test_forgetting_score_calculation(tmp_path: Path):
    storage = SQLiteStorage(tmp_path / "test.db")
    config = LifecyclePolicyConfig(
        forgetting_threshold=0.75,
        protected_importance=0.70,
        protected_access_count=3,
        recency_decay_lambda=0.05,
    )
    manager = MemoryLifecycleManager(storage, config)

    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # 1. Fresh, highly important memory
    fresh_mem = Memory(
        memory_id=str(uuid.uuid4()),
        user_id="user-1",
        content="Production cluster host is prod-cluster.internal",
        embedding=[0.1] * 384,
        created_at=now - timedelta(hours=2),
        last_accessed=now - timedelta(hours=1),
        access_count=5,
        importance_score=0.90,
        tier=WORKING,
    )
    score_fresh, breakdown_fresh = manager.calculate_forgetting_score(fresh_mem, now)
    # Fresh + high importance should have very low forgetting score
    assert score_fresh < 0.25
    assert breakdown_fresh.retention_score > 0.75

    # 2. Old, inactive, low-importance memory
    old_low_mem = Memory(
        memory_id=str(uuid.uuid4()),
        user_id="user-1",
        content="I drank coffee at 9am today",
        embedding=[0.1] * 384,
        created_at=now - timedelta(days=90),
        last_accessed=now - timedelta(days=85),
        access_count=0,
        importance_score=0.15,
        tier=ARCHIVE,
    )
    score_old, breakdown_old = manager.calculate_forgetting_score(old_low_mem, now)
    # Old + low importance should have high forgetting score
    assert score_old >= 0.75
    assert breakdown_old.retention_score < 0.25


def test_tier_transitions_working_to_long_term(tmp_path: Path):
    storage = SQLiteStorage(tmp_path / "test.db")
    config = LifecyclePolicyConfig(
        working_to_long_term_age=timedelta(days=7),
        working_to_long_term_importance_min=0.60,
        working_inactivity_threshold=timedelta(days=3),
    )
    manager = MemoryLifecycleManager(storage, config)

    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    # Aged working memory with sustained high importance and at least 1 access
    working_mem = Memory(
        memory_id=str(uuid.uuid4()),
        user_id="user-1",
        content="Database credentials and schema definitions for project Apollo",
        embedding=[0.1] * 384,
        created_at=now - timedelta(days=10),
        last_accessed=now - timedelta(days=5),  # inactive for 5 days (> 3 days threshold)
        access_count=2,
        importance_score=0.75,
        tier=WORKING,
    )
    storage.save_memory(working_mem)

    decision = manager.evaluate_memory(working_mem, now)
    assert decision.action == "TRANSITION"
    assert decision.from_tier == WORKING
    assert decision.to_tier == LONG_TERM


def test_frequently_accessed_memory_retention(tmp_path: Path):
    """Frequently accessed memories stay in active tiers longer even if older than the standard threshold."""
    storage = SQLiteStorage(tmp_path / "test.db")
    config = LifecyclePolicyConfig(
        working_to_long_term_age=timedelta(days=7),
        frequent_access_boost_threshold=3,
        working_inactivity_threshold=timedelta(days=3),
    )
    manager = MemoryLifecycleManager(storage, config)

    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    # Memory is 10 days old (> 7 days), BUT accessed frequently (4 times) and recently active (1 day ago)
    active_working_mem = Memory(
        memory_id=str(uuid.uuid4()),
        user_id="user-1",
        content="Current sprint active goal and ticket assignments",
        embedding=[0.1] * 384,
        created_at=now - timedelta(days=10),
        last_accessed=now - timedelta(days=1),  # recently active!
        access_count=4,                         # frequently accessed (>= 3)
        importance_score=0.65,
        tier=WORKING,
    )
    storage.save_memory(active_working_mem)

    decision = manager.evaluate_memory(active_working_mem, now)
    # Should NOT be demoted to LONG_TERM because of frequent active access
    assert decision.to_tier == WORKING


def test_old_low_value_memory_archival(tmp_path: Path):
    """SHORT_TERM memory demotes to ARCHIVE when older or low retention score."""
    storage = SQLiteStorage(tmp_path / "test.db")
    config = LifecyclePolicyConfig(
        short_term_to_archive_age=timedelta(days=30),
    )
    manager = MemoryLifecycleManager(storage, config)

    now = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)

    short_term_mem = Memory(
        memory_id=str(uuid.uuid4()),
        user_id="user-1",
        content="Temporary discussion notes on lunch venue options",
        embedding=[0.1] * 384,
        created_at=now - timedelta(days=40),
        last_accessed=now - timedelta(days=35),
        access_count=0,
        importance_score=0.35,
        tier=SHORT_TERM,
    )
    storage.save_memory(short_term_mem)

    decision = manager.evaluate_memory(short_term_mem, now)
    assert decision.action == "TRANSITION"
    assert decision.from_tier == SHORT_TERM
    assert decision.to_tier == ARCHIVE


def test_selective_forgetting_removes_obsolete_memory(tmp_path: Path):
    """Obsolete ARCHIVE memory with high forgetting score is selectively forgotten."""
    storage = SQLiteStorage(tmp_path / "test.db")
    config = LifecyclePolicyConfig(
        forgetting_threshold=0.75,
        archive_obsolete_age=timedelta(days=60),
    )
    manager = MemoryLifecycleManager(storage, config)

    now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)

    obsolete_mem = Memory(
        memory_id=str(uuid.uuid4()),
        user_id="user-1",
        content="The meeting was rescheduled to 4pm last month",
        embedding=[0.1] * 384,
        created_at=now - timedelta(days=70),
        last_accessed=now - timedelta(days=65),
        access_count=0,
        importance_score=0.18,
        tier=ARCHIVE,
    )
    storage.save_memory(obsolete_mem)

    report = manager.run_lifecycle_pass(now=now)
    assert report["forgotten_count"] == 1
    assert report["forgotten"][0]["memory_id"] == obsolete_mem.memory_id

    # Verify memory is deleted from active memories table
    assert storage.get_memory(obsolete_mem.memory_id) is None


def test_protection_of_high_importance_memories(tmp_path: Path):
    """High-importance memories (>= 0.70) are strictly protected against automatic forgetting."""
    storage = SQLiteStorage(tmp_path / "test.db")
    config = LifecyclePolicyConfig(
        forgetting_threshold=0.75,
        protected_importance=0.70,
        archive_obsolete_age=timedelta(days=60),
    )
    manager = MemoryLifecycleManager(storage, config)

    now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Even if stored in ARCHIVE and 100 days old, high importance must NOT be forgotten
    important_mem = Memory(
        memory_id=str(uuid.uuid4()),
        user_id="user-1",
        content="Core API architectural decision: Use SQLite WAL mode with microsecond timeouts",
        embedding=[0.1] * 384,
        created_at=now - timedelta(days=100),
        last_accessed=now - timedelta(days=95),
        access_count=1,
        importance_score=0.88,
        tier=ARCHIVE,
    )
    storage.save_memory(important_mem)

    report = manager.run_lifecycle_pass(now=now)
    assert report["forgotten_count"] == 0
    assert report["protected_count"] >= 1
    # Verify memory still exists
    assert storage.get_memory(important_mem.memory_id) is not None


def test_protection_of_frequently_accessed_memories(tmp_path: Path):
    """Frequently accessed memories (access_count >= 3) are protected from forgetting."""
    storage = SQLiteStorage(tmp_path / "test.db")
    config = LifecyclePolicyConfig(
        forgetting_threshold=0.75,
        protected_access_count=3,
        archive_obsolete_age=timedelta(days=60),
    )
    manager = MemoryLifecycleManager(storage, config)

    now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Low importance (0.30), but accessed 5 times
    frequent_mem = Memory(
        memory_id=str(uuid.uuid4()),
        user_id="user-1",
        content="Common bash aliases and terminal shortcuts used weekly",
        embedding=[0.1] * 384,
        created_at=now - timedelta(days=80),
        last_accessed=now - timedelta(days=70),
        access_count=5,
        importance_score=0.30,
        tier=ARCHIVE,
    )
    storage.save_memory(frequent_mem)

    report = manager.run_lifecycle_pass(now=now)
    assert report["forgotten_count"] == 0
    assert storage.get_memory(frequent_mem.memory_id) is not None


def test_auditable_forgetting_history(tmp_path: Path):
    """When a memory is forgotten, its history record with reason and content is preserved."""
    storage = SQLiteStorage(tmp_path / "test.db")
    config = LifecyclePolicyConfig(
        forgetting_threshold=0.75,
        archive_obsolete_age=timedelta(days=60),
    )
    manager = MemoryLifecycleManager(storage, config)

    now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)

    mem_id = str(uuid.uuid4())
    mem = Memory(
        memory_id=mem_id,
        user_id="user-1",
        content="Temporary wifi password was BlueSky2025",
        embedding=[0.1] * 384,
        created_at=now - timedelta(days=90),
        last_accessed=now - timedelta(days=90),
        access_count=0,
        importance_score=0.10,
        tier=ARCHIVE,
    )
    storage.save_memory(mem)

    report = manager.run_lifecycle_pass(now=now)
    assert report["forgotten_count"] == 1

    # Active memory is gone
    assert storage.get_memory(mem_id) is None

    # History audit record is PERMANENTLY PRESERVED
    history = storage.get_history(mem_id)
    assert len(history) >= 1
    forgotten_entry = history[-1]
    assert forgotten_entry["operation"] == "FORGOTTEN"
    assert forgotten_entry["old_content"] == "Temporary wifi password was BlueSky2025"
    assert "Obsolete archived memory" in forgotten_entry["reason"]

    # Metrics track forgotten memories
    metrics = storage.get_metrics("user-1")
    assert metrics["forgotten_memories"] == 1


def test_api_lifecycle_run_and_dry_run(tmp_path: Path):
    """Test POST /lifecycle/run endpoint with dry_run=True and dry_run=False."""
    db_path = tmp_path / "api_lifecycle.db"
    storage = SQLiteStorage(db_path)
    lifecycle_config = LifecyclePolicyConfig(
        forgetting_threshold=0.75,
        protected_importance=0.70,
        archive_obsolete_age=timedelta(days=60),
    )
    lifecycle_manager = MemoryLifecycleManager(storage, lifecycle_config)
    
    app.state.retrieval = RetrievalService(
        storage=storage,
        embeddings=EmbeddingService("all-MiniLM-L6-v2"),
        lifecycle_config=lifecycle_config,
        lifecycle_manager=lifecycle_manager,
    )

    now = utc_now()

    # Create an obsolete memory directly
    mem = Memory(
        memory_id=str(uuid.uuid4()),
        user_id="user-1",
        content="Outdated temporary note from last quarter",
        embedding=[0.1] * 384,
        created_at=now - timedelta(days=90),
        last_accessed=now - timedelta(days=85),
        access_count=0,
        importance_score=0.10,
        tier=ARCHIVE,
    )
    storage.save_memory(mem)

    with TestClient(app) as client:
        # 1. GET /lifecycle/policy
        resp_policy = client.get("/lifecycle/policy")
        assert resp_policy.status_code == 200
        policy_data = resp_policy.json()
        assert "forgetting_threshold" in policy_data
        assert "protected_importance" in policy_data

        # 2. POST /lifecycle/run with dry_run = True
        resp_dry = client.post("/lifecycle/run", json={"user_id": "user-1", "dry_run": True})
        assert resp_dry.status_code == 200
        dry_data = resp_dry.json()
        assert dry_data["dry_run"] is True
        assert dry_data["forgotten_count"] == 1
        # In dry run, memory must still be present in storage!
        assert storage.get_memory(mem.memory_id) is not None

        # 3. POST /lifecycle/run with dry_run = False
        resp_live = client.post("/lifecycle/run", json={"user_id": "user-1", "dry_run": False})
        assert resp_live.status_code == 200
        live_data = resp_live.json()
        assert live_data["dry_run"] is False
        assert live_data["forgotten_count"] == 1
        # Memory is now deleted from active store
        assert storage.get_memory(mem.memory_id) is None
