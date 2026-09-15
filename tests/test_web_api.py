import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.llm.client import CompressionResult, ConsolidationDecision, LLMClient
from app.memory.compression import CompressionConfig
from app.memory.consolidation import ConsolidationConfig
from app.memory.importance import HeuristicImportanceScorer
from app.memory.models import Memory
from app.memory.storage import SQLiteStorage
from app.memory.tiers import ARCHIVE, LONG_TERM, SHORT_TERM, WORKING, LifecyclePolicy
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retrieval import RetrievalService


class MockModel:
    def encode(self, text, normalize_embeddings=True):
        if "weather" in text.lower():
            return [0.0, 1.0]
        if "python" in text.lower():
            return [1.0, 0.0]
        if any(kw in text.lower() for kw in ["lsm", "lsm-trees", "memory-mapped", "throughput", "concurrency"]):
            return [-0.5, 0.5]  # ~0 cosine similarity with [0.5,0.5] - avoids DUPLICATE
        return [0.5, 0.5]


class MockLLM(LLMClient):
    def __init__(self):
        self.classify_calls = []
        self.compress_calls = []
        self.chat_calls = []

    def check_health(self) -> bool:
        return True

    def classify_memory(self, new_content: str, candidates: list[dict]) -> ConsolidationDecision:
        self.classify_calls.append((new_content, candidates))
        if candidates and candidates[0]["similarity"] > 0.8:
            return ConsolidationDecision(action="DUPLICATE", reason="exact duplicate fact")
        return ConsolidationDecision(action="NEW", reason="novel observation")

    def compress_memory(self, content: str, compression_level: int) -> CompressionResult:
        self.compress_calls.append((content, compression_level))
        return CompressionResult(
            compressed_content=f"Compact(L{compression_level}): {content}",
            reason="test compression",
        )

    def generate_chat_response(self, user_message: str, retrieved_memories=None, chat_history=None) -> str:
        self.chat_calls.append((user_message, retrieved_memories, chat_history))
        count = len(retrieved_memories) if retrieved_memories else 0
        return f"I processed your query using {count} retrieved memories."


def create_test_retrieval(tmp_path):
    storage = SQLiteStorage(tmp_path / "adam_web_test.db")
    embeddings = EmbeddingService("test-model", model=MockModel())
    scorer = HeuristicImportanceScorer()
    lifecycle_policy = LifecyclePolicy()
    llm = MockLLM()
    return RetrievalService(
        storage=storage,
        embeddings=embeddings,
        scorer=scorer,
        lifecycle_policy=lifecycle_policy,
        llm=llm,
        consolidation_config=ConsolidationConfig(candidate_limit=3, min_similarity=0.60),
        compression_config=CompressionConfig(working_to_long_term_level=1, short_term_to_archive_level=2),
    )


def test_chat_pipeline_turn_and_trace(tmp_path):
    app.state.retrieval = create_test_retrieval(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "user_id": "user-test",
            "message": "I love Python programming and building machine learning systems.",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "user_memory" in data
    assert "retrieved_memories" in data
    assert "assistant_memory" in data
    assert "pipeline_stages" in data

    # Check user memory attributes
    u_mem = data["user_memory"]
    assert u_mem["action"] == "NEW"
    assert u_mem["importance_score"] > 0.0
    assert u_mem["tier"] in ["WORKING", "SHORT_TERM"]
    assert u_mem["is_stored"] is True


def test_metrics_and_memory_listing_endpoints(tmp_path):
    app.state.retrieval = create_test_retrieval(tmp_path)
    client = TestClient(app)

    # Store initial memories
    client.post("/memory", json={"user_id": "user-1", "content": "I like python programming."})
    client.post("/memory", json={"user_id": "user-1", "content": "The weather is rainy."})

    # Test GET /metrics
    metrics_resp = client.get("/metrics?user_id=user-1")
    assert metrics_resp.status_code == 200
    metrics = metrics_resp.json()
    assert metrics["total_memories"] == 2
    assert "tier_counts" in metrics
    assert "avg_importance" in metrics

    # Test GET /memories
    memories_resp = client.get("/memories?user_id=user-1")
    assert memories_resp.status_code == 200
    memories = memories_resp.json()
    assert len(memories) == 2

    # Filter search
    search_resp = client.get("/memories?user_id=user-1&search=weather")
    assert search_resp.status_code == 200
    search_results = search_resp.json()
    assert len(search_results) == 1
    assert "weather" in search_results[0]["content"]


def test_memory_lifecycle_transition_and_history(tmp_path):
    app.state.retrieval = create_test_retrieval(tmp_path)
    client = TestClient(app)

    # Store a memory
    stored = client.post("/memory", json={"user_id": "user-1", "content": "Important research objective."}).json()
    memory_id = stored["memory_id"]

    # Transition memory to LONG_TERM
    trans_resp = client.post(
        f"/memory/{memory_id}/transition",
        json={"target_tier": "LONG_TERM"},
    )
    assert trans_resp.status_code == 200
    trans_data = trans_resp.json()
    assert trans_data["tier"] == "LONG_TERM"
    assert trans_data["compression_level"] == 1

    # Check History
    hist_resp = client.get(f"/memory/{memory_id}/history")
    assert hist_resp.status_code == 200
    history = hist_resp.json()
    assert len(history["history"]) >= 1
    assert history["history"][0]["operation"] == "COMPRESSED"

    # Global history
    all_hist_resp = client.get("/history")
    assert all_hist_resp.status_code == 200
    assert len(all_hist_resp.json()) >= 1


def test_system_status_and_database_reset(tmp_path):
    app.state.retrieval = create_test_retrieval(tmp_path)
    client = TestClient(app)

    # Seed
    client.post("/memory", json={"user_id": "user-1", "content": "Temp memory to clear."})
    assert app.state.retrieval.storage.count() == 1

    # System Status
    status_resp = client.get("/system/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert "ollama" in status_data
    assert "database" in status_data

    # Reset without confirm -> fails
    fail_reset = client.post("/reset", json={"confirm": False})
    assert fail_reset.status_code == 400

    # Reset with confirm -> succeeds
    ok_reset = client.post("/reset", json={"confirm": True})
    assert ok_reset.status_code == 200
    assert app.state.retrieval.storage.count() == 0


def test_static_index_and_assets_serving(tmp_path):
    app.state.retrieval = create_test_retrieval(tmp_path)
    client = TestClient(app)

    # Test root HTML serving
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "ADAM" in root_resp.text
    assert "Adaptive Dynamic AI Memory" in root_resp.text

    # Test static assets
    css_resp = client.get("/static/css/styles.css")
    assert css_resp.status_code == 200
    assert "tier-working" in css_resp.text

    js_resp = client.get("/static/js/app.js")
    assert js_resp.status_code == 200
    assert "handleChatSubmit" in js_resp.text


def test_technical_preferences_and_project_importance_scoring(tmp_path):
    app.state.retrieval = create_test_retrieval(tmp_path)
    client = TestClient(app)

    # 1. Test "I love dsa in java"
    res1 = client.post("/memory", json={"user_id": "user-1", "content": "I love dsa in java"}).json()
    assert res1["importance_score"] >= 0.70
    assert res1["tier"] == "WORKING"

    # 2. Test "i am currently doing a project of memory management"
    res2 = client.post("/memory", json={"user_id": "user-1", "content": "i am currently doing a project of memory management"}).json()
    assert res2["importance_score"] >= 0.70
    assert res2["tier"] == "WORKING"


def test_chat_pipeline_ignores_greetings_and_boilerplates(tmp_path):
    retrieval = create_test_retrieval(tmp_path)
    # Mock LLM returning boilerplate
    retrieval.llm.generate_chat_response = lambda user_message, retrieved_memories=None, chat_history=None: "Sure! How can I help you today?"
    app.state.retrieval = retrieval
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"user_id": "user-test", "message": "hello"},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["user_memory"]["is_stored"] is False
    assert data["user_memory"]["action"] == "FILLER"
    assert data["user_memory"]["memory"] is None

    assert data["assistant_memory"]["is_stored"] is False
    assert data["assistant_memory"]["action"] == "FILLER"
    assert data["assistant_memory"]["memory"] is None

    assert retrieval.storage.count() == 0


def test_chat_pipeline_stores_both_user_and_informative_llm_response(tmp_path):
    retrieval = create_test_retrieval(tmp_path)
    # Mock LLM returning informative technical fact
    retrieval.llm.generate_chat_response = lambda user_message, retrieved_memories=None, chat_history=None: (
        "In Rust, database engines use LSM-trees and memory-mapped files for high throughput and safe concurrency."
    )
    app.state.retrieval = retrieval
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"user_id": "user-test", "message": "I am building a database system in Rust"},
    )
    assert response.status_code == 200
    data = response.json()

    # User memory stored in WORKING
    assert data["user_memory"]["is_stored"] is True
    assert data["user_memory"]["tier"] == "WORKING"
    assert data["user_memory"]["memory"] is not None

    # Assistant informative response stored
    assert data["assistant_memory"]["is_stored"] is True
    assert data["assistant_memory"]["tier"] in ["WORKING", "SHORT_TERM"]
    assert data["assistant_memory"]["memory"] is not None

    # Both stored in database
    assert retrieval.storage.count() == 2


def test_chat_endpoint_stores_low_value_non_filler_in_archive(tmp_path):
    retrieval = create_test_retrieval(tmp_path)
    retrieval.llm.generate_chat_response = lambda user_message, retrieved_memories=None, chat_history=None: "Understood."
    app.state.retrieval = retrieval
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"user_id": "user-test", "message": "The room temperature is 21 degrees today"},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["user_memory"]["is_stored"] is True
    assert data["user_memory"]["tier"] == "ARCHIVE"
    assert data["user_memory"]["importance_score"] <= 0.30


def test_scorer_returns_detailed_signal_breakdown():
    from app.memory.importance import HeuristicImportanceScorer
    scorer = HeuristicImportanceScorer()

    breakdown = scorer.score_with_breakdown("Please remember that my primary language is Python.")
    assert breakdown["is_filler"] is False
    assert 0.0 < breakdown["total"] <= 1.0
    assert "signals" in breakdown
    for signal_name in ["intent", "specificity", "durability", "salience", "recurrence", "recency"]:
        assert signal_name in breakdown["signals"]
        sig = breakdown["signals"][signal_name]
        assert "value" in sig
        assert "weight" in sig
        assert "contribution" in sig
        assert "reason" in sig
        assert sig["contribution"] == round(sig["value"] * sig["weight"], 4)


def test_domain_agnostic_specificity_recognizes_non_cs_entities():
    from app.memory.importance import HeuristicImportanceScorer
    scorer = HeuristicImportanceScorer()

    # Non-CS medical content with specific entities and metrics
    medical = scorer.score_with_breakdown("Patient presented with HbA1c 7.8% and prescribed Metformin 500mg daily.")
    spec_signal = medical["signals"]["specificity"]
    assert spec_signal["value"] >= 0.45
    assert "entities" in spec_signal["reason"].lower() or "entity" in spec_signal["reason"].lower()


def test_heuristic_first_consolidation_skips_llm_for_exact_duplicate(tmp_path):
    from app.llm.client import ConsolidationDecision
    from tests.test_phase1 import build_phase3_service

    # Provide an LLM that would fail if called
    class StrictLLM:
        def classify_memory(self, new_content, candidates):
            raise AssertionError("LLM should not be called for heuristic DUPLICATE (sim >= 0.92)!")

    service, _ = build_phase3_service(tmp_path, [])
    service.consolidation.llm = StrictLLM()

    # Seed initial memory
    service.store_memory("user-1", "I use Python for analysis.")
    initial_count = service.storage.count()

    # Store identical memory (FakeModel returns [0.0, 1.0] for both -> cosine similarity = 1.0 >= 0.92)
    trace = service.store_memory_with_trace("user-1", "I use Python for analysis.")
    assert trace["action"] == "DUPLICATE"
    assert "Heuristic" in trace["decision_reason"]
    assert service.storage.count() == initial_count


def test_source_role_isolation_prevents_cross_role_merging(tmp_path):
    from app.llm.client import ConsolidationDecision
    from tests.test_phase1 import build_phase3_service

    service, _ = build_phase3_service(
        tmp_path,
        [ConsolidationDecision(
            action="RELATED",
            merged_content="Combined user and assistant content.",
            reason="related topic",
        )],
    )

    # Store user query
    user_mem = service.store_memory("user-1", "Can you explain memory management?", source_role="user")
    assert user_mem is not None

    # Simulate assistant response on the same topic — with role isolation it should not merge into user's query
    assistant_trace = service.store_memory_with_trace(
        "user-1",
        "Memory management involves tiering and adaptive decay.",
        source_role="assistant",
    )
    # Both memories should exist independently
    assert assistant_trace["memory"].memory_id != user_mem.memory_id
    assert service.storage.count() == 2


def test_safe_contradiction_preserves_both_memories_and_marks_superseded(tmp_path):
    from app.llm.client import ConsolidationDecision
    from tests.test_phase1 import build_phase3_service

    service, _ = build_phase3_service(
        tmp_path,
        [ConsolidationDecision(
            action="CONTRADICTORY",
            merged_content="I now use Rust exclusively.",
            reason="user switched technologies",
        )],
    )

    # Temporarily set high heuristic threshold so it reaches LLM branch
    service.consolidation.config = service.consolidation.config.__class__(
        candidate_limit=2,
        min_similarity=0.35,
        heuristic_duplicate_threshold=1.5,
        heuristic_new_threshold=0.0,
    )

    old_mem = service.store_memory("user-1", "I use Python for analysis.")
    new_mem = service.store_memory("user-1", "I now use Rust exclusively.")

    assert new_mem.memory_id != old_mem.memory_id
    # Old memory should have superseded_by set to new memory
    updated_old = service.storage.get_memory(old_mem.memory_id)
    assert updated_old.superseded_by == new_mem.memory_id
    # Both memories preserved in database
    assert service.storage.count() == 2




