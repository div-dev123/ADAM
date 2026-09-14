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
        consolidation_config=ConsolidationConfig(candidate_limit=3, min_similarity=0.35),
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


