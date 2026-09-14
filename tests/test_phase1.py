import json
import sqlite3
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.llm.client import CompressionResult, ConsolidationDecision
from app.memory.compression import CompressionConfig
from app.memory.consolidation import ConsolidationConfig
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


class FakeLLM:
    def __init__(self, decisions=None):
        self.decisions = list(decisions or [])
        self.classification_candidates = []
        self.compression_levels = []

    def classify_memory(self, new_content, candidates):
        self.classification_candidates.append(candidates)
        return self.decisions.pop(0)

    def compress_memory(self, content, compression_level):
        self.compression_levels.append(compression_level)
        return CompressionResult(
            compressed_content=f"Compressed[{compression_level}]: {content}",
            reason="deterministic test compression",
        )


def build_service(tmp_path):
    return RetrievalService(
        SQLiteStorage(tmp_path / "adam.db"),
        EmbeddingService("fake-model", model=FakeModel()),
    )


def build_phase3_service(tmp_path, decisions):
    llm = FakeLLM(decisions)
    service = RetrievalService(
        SQLiteStorage(tmp_path / "adam.db"),
        EmbeddingService("fake-model", model=FakeModel()),
        llm=llm,
        consolidation_config=ConsolidationConfig(
            candidate_limit=2, min_similarity=0.35
        ),
        compression_config=CompressionConfig(
            working_to_long_term_level=1,
            short_term_to_archive_level=2,
        ),
    )
    return service, llm


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

    assert health.json() == {"status": "ok", "phase": 3}
    assert stored.status_code == 201
    assert 0.0 <= stored.json()["importance_score"] <= 1.0
    assert stored.json()["tier"] == WORKING
    assert retrieved.status_code == 200
    assert "compression_level" in retrieved.json()["results"][0]["memory"]
    assert "updated_at" in retrieved.json()["results"][0]["memory"]


def test_new_memory_is_created_and_only_similar_candidates_are_sent(tmp_path):
    service, llm = build_phase3_service(
        tmp_path, [ConsolidationDecision(action="NEW", reason="new fact")]
    )

    memory = service.store_memory("user-1", "A completely new weather fact.")

    assert service.storage.count() == 1
    assert memory.tier in {WORKING, SHORT_TERM, ARCHIVE}
    assert llm.classification_candidates == [[]]


def test_duplicate_updates_access_without_creating_memory(tmp_path):
    seed = build_service(tmp_path)
    original = seed.store_memory("user-1", "I use Python for analysis.")
    service, llm = build_phase3_service(
        tmp_path, [ConsolidationDecision(action="DUPLICATE", reason="same fact")]
    )

    result = service.store_memory("user-1", "I use Python for analysis.")

    assert result.memory_id == original.memory_id
    assert service.storage.count() == 1
    assert result.access_count == 1
    assert service.storage.get_history(original.memory_id)[0]["operation"] == "DUPLICATE"
    assert len(llm.classification_candidates[0]) == 1


def test_related_merges_and_recalculates_embedding(tmp_path):
    seed = build_service(tmp_path)
    original = seed.store_memory("user-1", "I use Python for analysis.")
    service, _ = build_phase3_service(
        tmp_path,
        [ConsolidationDecision(
            action="RELATED",
            merged_content="I use Python and pandas for analysis.",
            reason="adds a tool",
        )],
    )

    result = service.store_memory("user-1", "I also use pandas for analysis.")

    assert result.memory_id == original.memory_id
    assert result.content == "I use Python and pandas for analysis."
    assert result.embedding == [0.0, 1.0]
    assert service.storage.get_history(result.memory_id)[0]["operation"] == "RELATED"


def test_contradictory_update_preserves_audit_history(tmp_path):
    seed = build_service(tmp_path)
    original = seed.store_memory("user-1", "I use Python for analysis.")
    service, _ = build_phase3_service(
        tmp_path,
        [ConsolidationDecision(
            action="CONTRADICTORY",
            merged_content="I now use Rust for systems work.",
            reason="newer preference",
        )],
    )

    result = service.store_memory("user-1", "I now use Rust for systems work.")
    history = service.storage.get_history(result.memory_id)

    assert result.memory_id == original.memory_id
    assert result.content == "I now use Rust for systems work."
    assert history[0]["old_content"] == "I use Python for analysis."
    assert history[0]["new_content"] == result.content


def test_working_to_long_term_compression_preserves_importance(tmp_path):
    service, llm = build_phase3_service(
        tmp_path, [ConsolidationDecision(action="NEW")]
    )
    memory = service.store_memory("user-1", "My goal is to pass AWS certification.")
    importance = memory.importance_score

    result = service.transition_memory(memory.memory_id, LONG_TERM)

    assert result.tier == LONG_TERM
    assert result.compression_level == 1
    assert result.importance_score == importance
    assert llm.compression_levels == [1]
    assert service.storage.get_history(memory.memory_id)[0]["operation"] == "COMPRESSED"


def test_short_term_to_archive_uses_stronger_compression(tmp_path):
    service, llm = build_phase3_service(
        tmp_path, [ConsolidationDecision(action="NEW")]
    )
    memory = service.store_memory("user-1", "A medium-value project detail.")
    memory.tier = SHORT_TERM
    service.storage.update_memory(memory)
    importance = memory.importance_score

    result = service.transition_memory(memory.memory_id, ARCHIVE)

    assert result.tier == ARCHIVE
    assert result.compression_level == 2
    assert result.importance_score == importance
    assert llm.compression_levels == [2]


def test_greetings_and_trivial_acknowledgements_produce_no_memory(tmp_path):
    service = build_service(tmp_path)
    
    greetings = ["hi", "hello", "hey there", "good morning", "howdy"]
    acknowledgements = ["ok", "okay", "thanks", "thank you", "cool", "bye", "sounds good"]

    for msg in greetings + acknowledgements:
        stored = service.store_memory("user-1", msg)
        assert stored is None
        trace = service.store_memory_with_trace("user-1", msg)
        assert trace["is_stored"] is False
        assert trace["action"] == "FILLER"
        assert trace["memory"] is None
        assert trace["importance_score"] == 0.0

    assert service.storage.count() == 0


def test_low_value_useful_information_enters_archive_directly(tmp_path):
    service = build_service(tmp_path)
    
    low_val_1 = service.store_memory("user-1", "The room temperature is 21 degrees today")
    low_val_2 = service.store_memory("user-1", "I had a sandwich for lunch.")

    assert low_val_1 is not None
    assert low_val_1.importance_score <= 0.30
    assert low_val_1.tier == ARCHIVE

    assert low_val_2 is not None
    assert low_val_2.importance_score <= 0.30
    assert low_val_2.tier == ARCHIVE

    assert service.storage.count() == 2


def test_medium_value_information_enters_short_term(tmp_path):
    service = build_service(tmp_path)
    
    msg = service.store_memory("user-1", "We discussed the sprint goals and assigned tasks to teammates.")
    assert msg is not None
    assert 0.30 < msg.importance_score < 0.70
    assert msg.tier == SHORT_TERM


def test_high_value_persistent_information_enters_working(tmp_path):
    service = build_service(tmp_path)
    
    mem1 = service.store_memory("user-1", "I love dsa in java")
    mem2 = service.store_memory("user-1", "i am currently doing a project of memory management")
    mem3 = service.store_memory("user-1", "My goal is to pass the AWS certification.")

    for mem in [mem1, mem2, mem3]:
        assert mem is not None
        assert mem.importance_score >= 0.70
        assert mem.tier == WORKING

