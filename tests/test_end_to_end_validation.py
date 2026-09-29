"""Full end-to-end integration and lifecycle validation for ADAM."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from unittest.mock import MagicMock
from app.integrations.langchain_memory import ADAMChatMessageHistory, ADAMLangChainMemory
from app.integrations.langgraph_tools import create_adam_tools, retrieve_memory, store_memory
from app.llm.client import CompressionResult, ConsolidationDecision, LLMClient, OllamaClient
from app.main import app
from app.memory.compression import CompressionConfig, CompressionService
from app.memory.consolidation import ConsolidationConfig, ConsolidationService
from app.memory.importance import HeuristicImportanceScorer, ImportanceWeights, is_filler
from app.memory.lifecycle import LifecyclePolicyConfig, MemoryLifecycleManager
from app.memory.models import Memory, utc_now
from app.memory.storage import SQLiteStorage, create_memory
from app.memory.tiers import ARCHIVE, LONG_TERM, SHORT_TERM, WORKING, TierAssigner
from app.retrieval.context_builder import ContextBudgetConfig, ContextBudgeter, ContextBuilder
from app.retrieval.drift import DriftConfig, QueryDriftDetector
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.hybrid import HybridSearchService
from app.retrieval.ranking import MultiSignalRanker, RankingWeights
from app.retrieval.retrieval import RetrievalService


class MockE2ELLM(LLMClient):
    """Deterministic LLM mock for end-to-end integration validation."""

    def __init__(self):
        self.last_prompt = None

    def check_health(self) -> bool:
        return True

    def classify_memory(self, new_content: str, candidates: list[dict]) -> ConsolidationDecision:
        return ConsolidationDecision(action="NEW", reason="Novel information")

    def compress_memory(self, content: str, compression_level: int) -> CompressionResult:
        return CompressionResult(
            compressed_content=f"[L{compression_level} Compressed] {content[:60]}...",
            reason="Deterministic test compression",
        )

    def generate_chat_response(
        self,
        user_message: str,
        retrieved_memories=None,
        chat_history=None,
        formatted_context=None,
    ) -> str:
        self.last_prompt = formatted_context
        ctx_summary = (
            f" Utilizing {len(retrieved_memories)} architectural memory references."
            if retrieved_memories
            else " Confirmed architecture blueprint."
        )
        return f"System configuration recorded: cluster cache state tracking updated.{ctx_summary}"


def setup_test_system(tmp_path: Path, llm: LLMClient = None):
    """Initializes a complete, isolated ADAM instance for testing."""
    storage = SQLiteStorage(tmp_path / "e2e_adam.db")
    embeddings = EmbeddingService("all-MiniLM-L6-v2")
    llm = llm or MockE2ELLM()
    scorer = HeuristicImportanceScorer(ImportanceWeights())
    tier_assigner = TierAssigner()

    compression_service = CompressionService(
        storage=storage,
        llm=llm,
        embeddings=embeddings,
        config=CompressionConfig(),
    )
    lifecycle_manager = MemoryLifecycleManager(
        storage=storage,
        config=LifecyclePolicyConfig(),
        compression_service=compression_service,
    )
    drift_detector = QueryDriftDetector(
        embeddings=embeddings,
        config=DriftConfig(),
    )
    ranking_weights = RankingWeights()
    ranker = MultiSignalRanker(weights=ranking_weights)
    budget_config = ContextBudgetConfig(token_budget=500, redundancy_threshold=0.85)
    budgeter = ContextBudgeter(config=budget_config)
    context_builder = ContextBuilder(budgeter=budgeter)

    retrieval_service = RetrievalService(
        storage=storage,
        embeddings=embeddings,
        scorer=scorer,
        tier_assigner=tier_assigner,
        lifecycle_manager=lifecycle_manager,
        drift_detector=drift_detector,
        ranking_weights=ranking_weights,
        ranker=ranker,
        budget_config=budget_config,
        budgeter=budgeter,
        context_builder=context_builder,
        llm=llm,
    )
    return storage, retrieval_service, lifecycle_manager, llm


def test_complete_end_to_end_lifecycle_scenario(tmp_path):
    """Validates the full lifecycle flow:
    store fact -> assistant response -> retrieve fact -> update/contradict ->
    consolidation history -> aging/compression -> retrieve after transition.
    """
    storage, service, lifecycle, llm = setup_test_system(tmp_path)
    user_id = "e2e_user"

    # Step 1: User stores an initial critical architecture decision
    user_msg_1 = "Please remember: we are building our distributed cache using Redis Cluster with 6 shards."
    turn_1 = service.chat_turn(user_id, user_msg_1)

    assert turn_1["response"] != ""
    assert turn_1["user_memory"]["is_stored"] is True
    # Important system architecture statement enters WORKING tier
    assert turn_1["user_memory"]["tier"] == WORKING
    assert turn_1["user_memory"]["importance_score"] >= 0.60
    initial_mem_id = turn_1["user_memory"]["memory_id"]

    # Step 2: Assistant response memory is analyzed and stored with source_role='assistant'
    assert turn_1["assistant_memory"]["is_stored"] is True
    assert turn_1["assistant_memory"]["memory_id"] is not None
    assistant_mem = storage.get_memory(turn_1["assistant_memory"]["memory_id"])
    assert assistant_mem.source_role == "assistant"

    # Step 3: Retrieval in a later query retrieves the initial fact with complete multi-signal metadata
    query_1 = "What cache architecture are we using?"
    retrieved_1 = service.search(user_id, query_1, top_k=5)
    assert len(retrieved_1) >= 1
    top_item = retrieved_1[0]
    assert "Redis Cluster" in top_item["memory"].content
    # Check multi-signal ranking fields
    assert "final_score" in top_item
    assert "signals" in top_item
    assert "selection_reason" in top_item
    assert top_item["signal_scores"]["semantic_similarity"] > 0.40

    # Step 4: Contradiction / Update
    # User contradicts the prior decision
    user_msg_2 = "Please remember: we quit Redis and switched to Memcached with 6 shards because of horizontal clustering."
    turn_2 = service.chat_turn(user_id, user_msg_2)

    # Verify consolidation heuristic detected CONTRADICTORY action
    assert turn_2["user_memory"]["is_stored"] is True
    assert turn_2["user_memory"]["action"] == "CONTRADICTORY"
    new_mem_id = turn_2["user_memory"]["memory_id"]

    # Step 5: Verify history preservation and superseding in SQLite
    old_mem = storage.get_memory(initial_mem_id)
    assert old_mem.superseded_by == new_mem_id
    history = storage.get_history(initial_mem_id)
    assert len(history) >= 1
    assert history[0]["operation"] == "CONTRADICTORY"
    assert "Memcached" in history[0]["new_content"]

    # Step 6: Retrieval after contradiction
    # Query must return the current Memcached fact and EXCLUDE the superseded Redis fact
    retrieved_2 = service.search(user_id, query_1, top_k=5)
    assert len(retrieved_2) >= 1
    memcached_item = next((item for item in retrieved_2 if item["memory"].memory_id == new_mem_id), None)
    assert memcached_item is not None
    assert "Memcached" in memcached_item["memory"].content
    # Ensure obsolete Redis memory is not returned
    retrieved_ids = [item["memory"].memory_id for item in retrieved_2]
    assert initial_mem_id not in retrieved_ids
    user_memories = [item for item in retrieved_2 if item["memory"].source_role == "user"]
    assert not any("Redis Cluster" in item["memory"].content for item in user_memories)

    # Step 7: Aging & Lifecycle Compression
    # Simulate time passing by 10 days on the active Memcached memory
    active_mem = storage.get_memory(new_mem_id)
    simulated_past = utc_now() - timedelta(days=10)
    active_mem.created_at = simulated_past
    active_mem.last_accessed = simulated_past
    storage.update_memory(active_mem)

    # Run lifecycle pass
    pass_report = lifecycle.run_lifecycle_pass(user_id=user_id, dry_run=False, now=utc_now())
    # Should transition WORKING -> LONG_TERM due to age >= 7 days with high importance
    assert pass_report["transition_count"] >= 1
    transition_ids = [t["memory_id"] for t in pass_report["transitions"]]
    assert new_mem_id in transition_ids

    updated_mem = storage.get_memory(new_mem_id)
    assert updated_mem.tier == LONG_TERM
    assert updated_mem.compression_level == 1
    assert "[L1 Compressed]" in updated_mem.content

    # Step 8: Retrieval after compression and tier migration
    retrieved_3 = service.search(user_id, query_1, scope_tiers=[WORKING, LONG_TERM], top_k=5)
    assert len(retrieved_3) >= 1
    memcached_item_3 = next((item for item in retrieved_3 if item["memory"].memory_id == new_mem_id), None)
    assert memcached_item_3 is not None
    assert memcached_item_3["memory"].tier == LONG_TERM
    assert "[L1 Compressed]" in memcached_item_3["memory"].content


def test_filler_and_boilerplate_rejection(tmp_path):
    """Verifies that greetings, acknowledgments, and assistant boilerplate are never stored."""
    storage, service, _, _ = setup_test_system(tmp_path)
    user_id = "filler_user"

    # User greetings
    for greeting in ["Hello there!", "Good morning!", "Thanks!", "Goodbye!"]:
        trace = service.store_memory_with_trace(user_id, greeting, source_role="user")
        assert trace["is_stored"] is False
        assert trace["action"] == "FILLER"

    # Verify storage remains empty
    assert len(storage.get_memories(user_id)) == 0

    # Assistant boilerplate check
    filler_res, reason = is_filler("Sure, I can help with that!")
    assert filler_res is True
    assert reason == "llm_boilerplate"

    filler_res2, reason2 = is_filler("As an AI language model, I do not have feelings.")
    assert filler_res2 is True
    assert reason2 == "llm_boilerplate"


def test_tier_assignment_and_anti_amnesia_protection(tmp_path):
    """Verifies initial tier boundaries and protection from accidental forgetting."""
    storage, service, lifecycle, _ = setup_test_system(tmp_path)
    user_id = "tier_user"

    # High-importance fact enters WORKING
    high_trace = service.store_memory_with_trace(
        user_id,
        "Remember that our cryptographic signing key for API auth is rotated every 30 days via HSM.",
        source_role="user",
    )
    assert high_trace["tier"] == WORKING
    assert high_trace["importance_score"] >= 0.70
    high_id = high_trace["memory"].memory_id

    # Low-importance casual note enters ARCHIVE (or SHORT_TERM)
    low_trace = service.store_memory_with_trace(
        user_id,
        "I noticed the room temperature was 71 degrees.",
        source_role="user",
    )
    assert low_trace["importance_score"] <= 0.35

    # Simulate 65 days of inactivity on both memories
    stale_time = utc_now() - timedelta(days=65)
    high_mem = storage.get_memory(high_id)
    high_mem.created_at = stale_time
    high_mem.last_accessed = stale_time
    storage.update_memory(high_mem)

    # Evaluate lifecycle forgetting
    pass_report = lifecycle.run_lifecycle_pass(user_id=user_id, dry_run=True, now=utc_now())
    protected_ids = [p["memory_id"] for p in pass_report["protected"]]
    forgotten_ids = [f["memory_id"] for f in pass_report["forgotten"]]

    # Anti-amnesia protection: High-importance memories MUST NEVER be forgotten
    assert high_id in protected_ids
    assert high_id not in forgotten_ids


def test_langchain_and_langgraph_integration_e2e(tmp_path):
    """Validates that ADAM functions seamlessly with LangChain memory and LangGraph agent tools."""
    user_id = "langchain_user"
    mock_client = MagicMock()
    mock_client.post.return_value.status_code = 200
    mock_client.post.return_value.json.return_value = {
        "results": [
            {
                "similarity": 0.92,
                "memory_tier": "WORKING",
                "importance": 0.88,
                "memory": {"content": "We use Apache Kafka for streaming order events."},
            }
        ],
        "is_stored": True,
        "memory_id": "mem-kafka-123",
        "tier": "WORKING",
        "importance_score": 0.88,
    }

    # 1. LangChain Memory integration
    lc_memory = ADAMLangChainMemory(user_id=user_id, client=mock_client, return_messages=True)
    lc_memory.save_context(
        inputs={"input": "We use Apache Kafka for streaming order events."},
        outputs={"output": "Noted, Kafka is used for streaming orders."},
    )

    variables = lc_memory.load_memory_variables({"input": "What streaming broker do we use?"})
    assert "history" in variables
    assert len(variables["history"]) >= 2
    assert "Kafka" in variables["history"][0].content
    assert "relevant_memories" in variables
    assert len(variables["relevant_memories"]) >= 1

    # 2. LangGraph Tools integration
    tools = create_adam_tools(client=mock_client, user_id=user_id)
    assert len(tools) == 2
    tool_map = {t.name: t for t in tools}
    assert "retrieve_memory" in tool_map
    assert "store_memory" in tool_map

    # Tool invocation: Store
    store_res = tool_map["store_memory"].invoke({"content": "PostgreSQL replication lag threshold is set to 100ms."})
    assert "Memory successfully stored in ADAM" in store_res

    # Tool invocation: Retrieve
    ret_res = tool_map["retrieve_memory"].invoke({"query": "What is our streaming broker?"})
    assert "We use Apache Kafka" in ret_res


def test_real_local_ollama_conversation_turn():
    """Validates real local Ollama execution with qwen2.5:3b when available."""
    client = OllamaClient("http://localhost:11434", "qwen2.5:3b", timeout=15.0)
    if not client.check_health():
        pytest.skip("Local Ollama server is not running or unreachable")

    # Injected context block
    context = (
        "=== RETRIEVED MEMORIES (Ranked & Budget-Filtered) ===\n"
        "• [WORKING | Importance: 0.95 | Score: 0.92] The user's production server is located at 192.168.1.50."
    )
    prompt = "What is the IP address of my production server? Answer with just the IP address."
    response = client.generate_chat_response(
        user_message=prompt,
        formatted_context=context,
    )
    assert "192.168.1.50" in response


def test_frontend_visible_api_endpoints_real_data(tmp_path):
    """Verifies that the existing web interface endpoints return real, well-formed data."""
    with TestClient(app) as test_client:
        # 1. UI Root
        resp_root = test_client.get("/")
        assert resp_root.status_code == 200
        assert "ADAM" in resp_root.text

        # 2. System Status
        resp_status = test_client.get("/system/status")
        assert resp_status.status_code == 200
        status_data = resp_status.json()
        assert "database" in status_data
        assert "total_memories" in status_data["database"]
        assert "config" in status_data

        # 3. Chat Endpoint
        chat_payload = {
            "user_id": "api_eval_user",
            "message": "We deploy our containerized microservices to Google Cloud Run.",
            "top_k": 3,
            "token_budget": 400,
        }
        resp_chat = test_client.post("/chat", json=chat_payload)
        assert resp_chat.status_code == 200
        chat_data = resp_chat.json()
        assert "pipeline_stages" in chat_data
        assert "user_memory" in chat_data
        assert "context_budget" in chat_data
        assert chat_data["user_memory"]["is_stored"] is True
        assert "Google Cloud Run" in chat_data["user_memory"]["content"]

        # 4. Memories List
        resp_mems = test_client.get("/memories?user_id=api_eval_user")
        assert resp_mems.status_code == 200
        mems = resp_mems.json()
        assert len(mems) >= 1
        assert any("Cloud Run" in m["content"] for m in mems)

        # 5. Retrieve Endpoint
        retrieve_payload = {
            "user_id": "api_eval_user",
            "query": "Where do we deploy microservices?",
            "top_k": 3,
            "token_budget": 400,
        }
        resp_ret = test_client.post("/retrieve", json=retrieve_payload)
        assert resp_ret.status_code == 200
        ret_data = resp_ret.json()
        assert "results" in ret_data
        assert "context_budget" in ret_data
        assert len(ret_data["results"]) >= 1
        top_mem = ret_data["results"][0]
        assert "Cloud Run" in top_mem["memory"]["content"]
        assert "final_score" in top_mem
        assert "signal_scores" in top_mem
        assert "selection_reason" in top_mem

        # 6. Lifecycle Endpoints
        resp_lifecycle = test_client.post("/lifecycle/run", json={"user_id": "api_eval_user", "dry_run": True})
        assert resp_lifecycle.status_code == 200
        lifecycle_data = resp_lifecycle.json()
        assert "transitions" in lifecycle_data
        assert "protected" in lifecycle_data

        resp_policy = test_client.get("/lifecycle/policy")
        assert resp_policy.status_code == 200
        assert "forgetting_threshold" in resp_policy.json()

        # 7. System Metrics & History
        resp_history = test_client.get("/history?user_id=api_eval_user")
        assert resp_history.status_code == 200
        assert isinstance(resp_history.json(), list)

        resp_metrics = test_client.get("/metrics")
        assert resp_metrics.status_code == 200
        assert "total_memories" in resp_metrics.json()
