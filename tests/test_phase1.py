import json
import sqlite3
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.memory.importance import HeuristicImportanceScorer
from app.memory.models import Memory
from app.memory.storage import SQLiteStorage, create_memory
from app.memory.tiers import ARCHIVE, LONG_TERM, SHORT_TERM, LifecyclePolicy, WORKING
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retrieval import RetrievalService


class FakeModel:
    def encode(self, text, normalize_embeddings=True):
        assert normalize_embeddings is True
        if "weather" in text.lower():
            return [0.0, 1.0]
        if text == "What is ADAM?":
            return [0.95, 0.05]
        if text.startswith("The project is called ADAM"):
            return [1.0, 0.0]
        if text.startswith("ADAM uses semantic"):
            return [0.8, 0.2]
        return [0.0, 1.0]


def build_service(tmp_path):
    return RetrievalService(
        SQLiteStorage(tmp_path / "adam.db"),
        EmbeddingService("fake-model", model=FakeModel()),
    )


def test_database_initialization_and_memory_persistence(tmp_path):
    database_path = tmp_path / "data" / "adam.db"
    storage = SQLiteStorage(database_path)
    memory = create_memory("user-1", "A stored memory.", [1.0, 0.0])

    storage.save_memory(memory)
    persisted = storage.get_memories("user-1")[0]

    assert database_path.exists()
    assert storage.count() == 1
    assert persisted.memory_id == memory.memory_id
    assert persisted.embedding == [1.0, 0.0]
    assert persisted.updated_at == persisted.created_at

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT embedding, compression_level, updated_at FROM memories WHERE memory_id = ?",
            (memory.memory_id,),
        ).fetchone()
    assert json.loads(row[0]) == [1.0, 0.0]
    assert row[1] == 0
    assert row[2] == memory.created_at.isoformat()


def test_importance_scoring_is_independent_from_lifecycle_policy():
    scorer = HeuristicImportanceScorer()
    policy = LifecyclePolicy()
    score = scorer.score("My goal is to pass the AWS certification.")

    assert 0.0 <= score <= 1.0
    assert policy.initial_tier(score) == WORKING
    assert policy.transition(
        Memory("m", "user-1", "x", [1.0], datetime.now(timezone.utc), datetime.now(timezone.utc),
               access_count=0, importance_score=score, tier=WORKING)
    ) == WORKING


def test_initial_tier_policy_allows_low_value_archive_and_high_value_working():
    policy = LifecyclePolicy()

    assert policy.initial_tier(0.10) == ARCHIVE
    assert policy.initial_tier(0.40) == SHORT_TERM
    assert policy.initial_tier(0.90) == WORKING


def test_working_moves_to_long_term_without_changing_importance():
    policy = LifecyclePolicy(working_to_long_term_age=timedelta(days=7))
    created = datetime(2026, 1, 1, tzinfo=timezone.utc)
    memory = Memory(
        "m", "user-1", "valuable", [1.0], created, created,
        access_count=2, importance_score=0.91, tier=WORKING,
    )

    policy.apply_transition(memory, created + timedelta(days=8))

    assert memory.tier == LONG_TERM
    assert memory.importance_score == 0.91


def test_short_term_moves_to_archive_when_compressed():
    policy = LifecyclePolicy()
    created = datetime(2026, 1, 1, tzinfo=timezone.utc)
    memory = Memory(
        "m", "user-1", "temporary", [1.0], created, created,
        importance_score=0.60, tier=SHORT_TERM, compression_level=1,
    )

    policy.apply_transition(memory, created + timedelta(days=1))

    assert memory.tier == ARCHIVE
    assert memory.importance_score == 0.60


def test_lifecycle_metadata_update_persists(tmp_path):
    storage = SQLiteStorage(tmp_path / "adam.db")
    memory = create_memory("user-1", "A memory.", [1.0, 0.0])
    memory.importance_score = 0.8
    memory.tier = LONG_TERM
    memory.compression_level = 1
    memory.updated_at = memory.created_at + timedelta(days=1)
    storage.save_memory(memory)

    memory.tier = ARCHIVE
    memory.updated_at = memory.updated_at + timedelta(days=1)
    storage.update_memory(memory)
    persisted = storage.get_memories("user-1")[0]

    assert persisted.importance_score == 0.8
    assert persisted.tier == ARCHIVE
    assert persisted.compression_level == 1
    assert persisted.updated_at == memory.updated_at


def test_semantic_retrieval_still_ranks_relevant_memories_and_tracks_access(tmp_path):
    service = build_service(tmp_path)
    service.store_memory("user-1", "The project is called ADAM and focuses on adaptive memory management.")
    service.store_memory("user-1", "ADAM uses semantic memory retrieval to find relevant information.")
    service.store_memory("user-1", "The weather today is sunny.")

    results = service.search("user-1", "What is ADAM?", top_k=2)

    assert len(results) == 2
    assert results[0]["memory"].content.startswith("The project is called ADAM")
    assert results[0]["similarity"] > results[1]["similarity"]
    assert results[0]["memory"].access_count == 1


def test_api_health_and_memory_endpoints(tmp_path):
    app.state.retrieval = build_service(tmp_path)

    with TestClient(app) as client:
        health = client.get("/health")
        stored = client.post(
            "/memory",
            json={"user_id": "user-1", "content": "My goal is to pass AWS certification."},
        )
        retrieved = client.post(
            "/retrieve",
            json={"user_id": "user-1", "query": "What is my goal?", "top_k": 1},
        )

    assert health.json() == {"status": "ok", "phase": 2}
    assert stored.status_code == 201
    assert 0.0 <= stored.json()["importance_score"] <= 1.0
    assert stored.json()["tier"] == WORKING
    assert retrieved.status_code == 200
    assert "compression_level" in retrieved.json()["results"][0]["memory"]
    assert "updated_at" in retrieved.json()["results"][0]["memory"]
