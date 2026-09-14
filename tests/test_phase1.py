import json
import sqlite3
import pytest

from fastapi.testclient import TestClient

from app.main import app
from app.llm.client import ConsolidationDecision
from app.memory.consolidation import ConsolidationConfig
from app.memory.importance import HeuristicImportanceScorer
from app.memory.models import Memory
from app.memory.storage import SQLiteStorage, create_memory
from app.memory.tiers import ARCHIVE, LONG_TERM, SHORT_TERM, TierAssigner, WORKING
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retrieval import RetrievalService


class FakeModel:
    def encode(self, text, normalize_embeddings=True):
        assert normalize_embeddings is True
        if "unrelated" in text.lower() or "weather" in text.lower():
            return [0.0, 1.0]
        if text == "What is ADAM?":
            return [0.95, 0.05]
        if text.startswith("The project is called ADAM"):
            return [1.0, 0.0]
        if text.startswith("ADAM uses semantic"):
            return [0.8, 0.2]
        return [0.0, 1.0]


class FakeLLM:
    def __init__(self, decision):
        self.decision = decision
        self.candidate_batches = []

    def analyze_consolidation(self, new_content, candidates):
        self.candidate_batches.append(candidates)
        return self.decision


def build_service(tmp_path):
    return RetrievalService(
        SQLiteStorage(tmp_path / "adam.db"),
        EmbeddingService("fake-model", model=FakeModel()),
    )


def build_consolidating_service(tmp_path, decision, candidate_limit=3):
    llm = FakeLLM(decision)
    service = RetrievalService(
        SQLiteStorage(tmp_path / "adam.db"),
        EmbeddingService("fake-model", model=FakeModel()),
        llm=llm,
        consolidation_config=ConsolidationConfig(
            candidate_limit=candidate_limit,
            min_similarity=0.35,
        ),
    )
    return service, llm


def test_database_initialization_and_memory_persistence(tmp_path):
    database_path = tmp_path / "data" / "adam.db"
    storage = SQLiteStorage(database_path)
    memory = create_memory("user-1", "A stored memory.", [1.0, 0.0])

    storage.save_memory(memory)
    persisted = storage.get_memories("user-1")

    assert database_path.exists()
    assert storage.count() == 1
    assert persisted[0].memory_id == memory.memory_id
    assert persisted[0].embedding == [1.0, 0.0]

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT embedding FROM memories WHERE memory_id = ?",
            (memory.memory_id,),
        ).fetchone()
    assert json.loads(row[0]) == [1.0, 0.0]


def test_embedding_generation_loads_configured_model():
    embedding_service = EmbeddingService("fake-model", model=FakeModel())

    vector = embedding_service.encode("ADAM memory")

    assert vector == [0.0, 1.0]
    assert embedding_service._model is not None


def test_structured_decision_requires_merged_content_for_updates():
    with pytest.raises(ValueError):
        ConsolidationDecision(action="RELATED").require_merged_content()

    decision = ConsolidationDecision(
        action="RELATED", merged_content="Merged memory."
    )
    assert decision.require_merged_content() == "Merged memory."


def test_importance_scoring_is_bounded_and_rewards_persistent_language():
    scorer = HeuristicImportanceScorer()

    persistent = scorer.score("My goal is to pass the AWS certification.")
    filler = scorer.score("Okay, thanks.")

    assert 0.0 <= filler <= 1.0
    assert 0.0 <= persistent <= 1.0
    assert persistent > filler


def test_tier_assignment_uses_configurable_thresholds():
    assigner = TierAssigner()

    assert assigner.assign(0.10) == ARCHIVE
    assert assigner.assign(0.30) == WORKING
    assert assigner.assign(0.60) == SHORT_TERM
    assert assigner.assign(0.90) == LONG_TERM


def test_semantic_retrieval_ranks_relevant_memories_and_tracks_access(tmp_path):
    service = build_service(tmp_path)
    service.store_memory(
        "user-1",
        "The project is called ADAM and focuses on adaptive memory management.",
    )
    service.store_memory(
        "user-1",
        "ADAM uses semantic memory retrieval to find relevant information.",
    )
    service.store_memory("user-1", "The weather today is sunny.")
    service.store_memory("other-user", "ADAM belongs to another user.")
    before_access = service.storage.get_memories("user-1")[0].last_accessed

    results = service.search("user-1", "What is ADAM?", top_k=2)

    assert len(results) == 2
    assert all(result["memory"].user_id == "user-1" for result in results)
    assert results[0]["memory"].content.startswith("The project is called ADAM")
    assert results[0]["similarity"] > results[1]["similarity"]
    assert results[0]["memory"].access_count == 1
    persisted = {
        memory.memory_id: memory
        for memory in service.storage.get_memories("user-1")
    }
    returned_id = results[0]["memory"].memory_id
    assert persisted[returned_id].access_count == 1
    assert persisted[returned_id].last_accessed >= before_access


def test_importance_and_tier_persist_in_sqlite(tmp_path):
    service = build_service(tmp_path)
    memory = service.store_memory(
        "user-1", "My goal is to pass the AWS certification."
    )

    persisted = service.storage.get_memories("user-1")[0]

    assert persisted.memory_id == memory.memory_id
    assert persisted.importance_score == memory.importance_score
    assert persisted.importance_score > 0.0
    assert persisted.tier == memory.tier


def test_duplicate_does_not_create_memory_and_updates_access(tmp_path):
    seed = build_service(tmp_path)
    original = seed.store_memory("user-1", "I use Python for data analysis.")
    service, llm = build_consolidating_service(
        tmp_path,
        ConsolidationDecision(action="DUPLICATE", reason="Same fact"),
    )

    result = service.store_memory("user-1", "I use Python for data analysis.")

    assert result.memory_id == original.memory_id
    assert service.storage.count() == 1
    assert result.access_count == 1
    assert len(llm.candidate_batches[0]) == 1
    assert service.storage.get_consolidation_events("user-1")[0]["action"] == "DUPLICATE"


@pytest.mark.parametrize(
    ("action", "incoming", "merged"),
    [
        ("RELATED", "I use Python and pandas.", "I use Python and pandas for data analysis."),
        ("CONTRADICTORY", "I now use Rust for systems work.", "I now use Rust for systems work."),
    ],
)
def test_related_and_contradictory_update_existing_memory(
    tmp_path, action, incoming, merged
):
    seed = build_service(tmp_path)
    original = seed.store_memory("user-1", "I use Python for data analysis.")
    service, _ = build_consolidating_service(
        tmp_path,
        ConsolidationDecision(
            action=action, merged_content=merged, reason="Test decision"
        ),
    )

    result = service.store_memory("user-1", incoming)
    persisted = service.storage.get_memories("user-1")
    events = service.storage.get_consolidation_events("user-1")

    assert result.memory_id == original.memory_id
    assert result.content == merged
    assert len(persisted) == 1
    assert result.access_count == 1
    assert events[0]["action"] == action
    assert events[0]["old_content"] == original.content


def test_new_unrelated_memory_is_stored_without_candidates(tmp_path):
    service, llm = build_consolidating_service(
        tmp_path,
        ConsolidationDecision(action="NEW/UNRELATED", reason="No match"),
    )

    result = service.store_memory("user-1", "The weather today is sunny.")

    assert service.storage.count() == 1
    assert result.content == "The weather today is sunny."
    assert llm.candidate_batches == [[]]


def test_consolidation_candidate_count_is_bounded(tmp_path):
    seed = build_service(tmp_path)
    for index in range(5):
        seed.store_memory("user-1", f"ADAM project fact {index}.")
    service, llm = build_consolidating_service(
        tmp_path,
        ConsolidationDecision(action="NEW/UNRELATED"),
        candidate_limit=2,
    )

    service.store_memory("user-1", "ADAM project update.")

    assert len(llm.candidate_batches[0]) == 2


def test_api_health_endpoint(tmp_path):
    app.state.retrieval = build_service(tmp_path)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "phase": 3}


def test_api_memory_and_retrieve_endpoints(tmp_path):
    app.state.retrieval = build_service(tmp_path)

    with TestClient(app) as client:
        stored = client.post(
            "/memory",
            json={
                "user_id": "user-1",
                "content": "The project is called ADAM and focuses on adaptive memory management.",
            },
        )
        retrieved = client.post(
            "/retrieve",
            json={"user_id": "user-1", "query": "What is ADAM?", "top_k": 1},
        )

    assert stored.status_code == 201
    assert 0.0 <= stored.json()["importance_score"] <= 1.0
    assert stored.json()["tier"] in {
        "WORKING", "SHORT_TERM", "LONG_TERM", "ARCHIVE"
    }
    assert retrieved.status_code == 200
    assert retrieved.json()["results"][0]["memory"]["content"].startswith(
        "The project is called ADAM"
    )
    assert retrieved.json()["results"][0]["memory"]["access_count"] == 1
    assert "importance_score" in retrieved.json()["results"][0]["memory"]
    assert "tier" in retrieved.json()["results"][0]["memory"]
