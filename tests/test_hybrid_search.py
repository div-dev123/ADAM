"""Tests and empirical performance benchmark for Hybrid Search (Dense + BM25) and RRF."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.memory.models import Memory, utc_now
from app.memory.storage import SQLiteStorage
from app.retrieval.bm25 import BM25Index, tokenize
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.hybrid import HybridSearchService, reciprocal_rank_fusion
from app.retrieval.retrieval import RetrievalService


# ===========================================================================
# 1. BM25 Unit Tests
# ===========================================================================

def test_bm25_tokenization():
    text = "FastAPI v0.110.0 with PostgreSQL-15 & OAuth2_Token!"
    tokens = tokenize(text)
    assert "fastapi" in tokens
    assert "v0.110.0" in tokens
    assert "postgresql-15" in tokens
    assert "oauth2_token" in tokens


def test_bm25_empty_corpus():
    index = BM25Index([])
    assert index.doc_count == 0
    assert index.search("fastapi") == []
    assert index.score("fastapi", 0) == 0.0


def test_bm25_exact_keyword_ranking():
    docs = [
        "Configured PostgreSQL database on port 5432 with replica.",
        "Configured Redis cache on port 6379 for session management.",
        "General database cluster architecture and backup strategy.",
    ]
    index = BM25Index(docs)
    results = index.search("postgresql 5432")

    # Document 0 should be top ranked because both "postgresql" and "5432" match
    assert results[0][1] == 0
    assert results[0][0] > results[1][0]
    # Document 1 matches "configured", "port", "on" but not "postgresql" or "5432"
    assert results[0][0] > results[2][0]


def test_bm25_length_normalization():
    # A short doc matching a term should score higher than a long doc with many other words
    docs = [
        "Python FastAPI",
        "This is an extraordinarily long text discussing numerous unrelated programming languages, frameworks, deployment strategies, and databases but also mentions Python.",
    ]
    index = BM25Index(docs)
    results = index.search("Python")
    assert results[0][1] == 0  # Short doc wins due to length normalization


# ===========================================================================
# 2. Reciprocal Rank Fusion (RRF) Unit Tests
# ===========================================================================

def _make_dummy_memory(mem_id: str, content: str) -> Memory:
    now = utc_now()
    return Memory(
        memory_id=mem_id,
        user_id="user-1",
        content=content,
        embedding=[0.1, 0.2],
        created_at=now,
        last_accessed=now,
    )


def test_rrf_consensus_promotion():
    """A document ranked #2 in BOTH Dense and BM25 should beat a document ranked #1 in Dense but unranked in BM25."""
    mem_a = _make_dummy_memory("A", "Memory A")
    mem_b = _make_dummy_memory("B", "Memory B")

    # Dense ranking: A is #1, B is #2
    dense_ranked = [(0.95, mem_a), (0.85, mem_b)]
    # BM25 ranking: B is #1, A is not ranked (score 0)
    bm25_ranked = [(12.0, mem_b)]

    fused = reciprocal_rank_fusion(dense_ranked, bm25_ranked, k=60)

    # RRF score for B: 1/(60+2) + 1/(60+1) = 0.016129 + 0.016393 = ~0.03252
    # RRF score for A: 1/(60+1) + 0        = 0.016393
    assert fused[0]["memory"].memory_id == "B"
    assert fused[1]["memory"].memory_id == "A"
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


def test_rrf_handles_disjoint_lists():
    mem_a = _make_dummy_memory("A", "Memory A")
    mem_b = _make_dummy_memory("B", "Memory B")

    dense_ranked = [(0.90, mem_a)]
    bm25_ranked = [(8.5, mem_b)]

    fused = reciprocal_rank_fusion(dense_ranked, bm25_ranked, k=60)
    assert len(fused) == 2
    ids = {item["memory"].memory_id for item in fused}
    assert ids == {"A", "B"}


# ===========================================================================
# 3. HybridSearchService API & Modes Tests
# ===========================================================================

def test_hybrid_search_modes(tmp_path):
    storage = SQLiteStorage(tmp_path / "test_modes.db")
    embeddings = EmbeddingService("all-MiniLM-L6-v2")
    service = RetrievalService(storage=storage, embeddings=embeddings)

    service.store_memory("u1", "FastAPI web framework with Python.")
    service.store_memory("u1", "Django web framework with Python.")
    service.store_memory("u1", "The weather is very sunny today.")

    # Test explicit mode flags
    hybrid_res = service.search("u1", "FastAPI", top_k=2, mode="hybrid")
    dense_res = service.search("u1", "FastAPI", top_k=2, mode="dense")
    sparse_res = service.search("u1", "FastAPI", top_k=2, mode="sparse")

    assert len(hybrid_res) == 2
    assert "FastAPI" in hybrid_res[0]["memory"].content

    assert len(dense_res) == 2
    assert "dense_score" in dense_res[0]

    assert len(sparse_res) == 2
    assert "bm25_score" in sparse_res[0]
    assert "FastAPI" in sparse_res[0]["memory"].content


def test_api_retrieve_returns_hybrid_telemetry(tmp_path):
    storage = SQLiteStorage(tmp_path / "test_api_hybrid.db")
    embeddings = EmbeddingService("all-MiniLM-L6-v2")
    app.state.retrieval = RetrievalService(storage=storage, embeddings=embeddings)

    with TestClient(app) as client:
        client.post("/memory", json={"user_id": "u-api", "content": "PostgreSQL database running on port 5432."})

        resp = client.post(
            "/retrieve",
            json={"user_id": "u-api", "query": "PostgreSQL port", "top_k": 1, "mode": "hybrid"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        top_hit = data["results"][0]
        assert "similarity" in top_hit
        assert "dense_score" in top_hit
        assert "bm25_score" in top_hit
        assert "rrf_score" in top_hit


# ===========================================================================
# 4. Empirical Performance Benchmark: Hybrid vs. Dense vs. BM25
# ===========================================================================

def test_hybrid_search_benchmark_proves_superiority(tmp_path):
    """Empirical evaluation benchmark comparing Dense-Only, BM25-Only, and Hybrid (RRF) search.

    Tests across 6 diverse query types:
    - Exact technical entity & port matching
    - Acronym / protocol matching
    - Conceptual synonymy (vocabulary mismatch)
    - Specific version numbers
    - Mixed semantic + exact keyword intent
    """
    storage = SQLiteStorage(tmp_path / "test_benchmark.db")
    embeddings = EmbeddingService("all-MiniLM-L6-v2")
    retrieval = RetrievalService(storage=storage, embeddings=embeddings)

    user_id = "bench-user"

    # Corpus of 8 realistic agent memories
    memories_data = [
        # Doc 0: Relational DB with exact port
        "Configured PostgreSQL database on port 5432 with read-replica.",
        # Doc 1: Cache with exact port (Distractor for Doc 0)
        "Configured Redis cache on port 6379 for fast session management.",
        # Doc 2: Generic relational cluster (Semantic sibling for Doc 0)
        "Relational SQL database cluster running across primary and secondary nodes.",
        # Doc 3: Acronym security protocol
        "Implemented OAuth2 PKCE authorization flow for secure mobile client login.",
        # Doc 4: Generic auth (Semantic sibling for Doc 3)
        "User authentication and single sign-on security system.",
        # Doc 5: Conceptual dislike (monolith)
        "The developer strongly dislikes monolithic application architecture.",
        # Doc 6: Microservices (Semantic counterpart for Doc 5)
        "Microservices architecture deployed with Kubernetes and Docker containers.",
        # Doc 7: Specific library and version
        "Upgraded machine learning pipeline to PyTorch v2.3 with CUDA 12 support.",
    ]

    for text in memories_data:
        retrieval.store_memory(user_id, text)

    # Benchmark test suite demonstrating lexical vs. semantic trade-offs
    benchmark_queries = [
        (
            "postgresql port 5432",
            "PostgreSQL database on port 5432",
            "Exact Entity & Port Number (BM25 & Hybrid hit #1)",
        ),
        (
            "OAuth2 PKCE authorization",
            "OAuth2 PKCE authorization",
            "Technical Acronym & Protocol (BM25 & Hybrid hit #1)",
        ),
        (
            "Which software design pattern is detested?",
            "dislikes monolithic application",
            "Zero Word Overlap Semantic Query (BM25 fails completely; Dense & Hybrid hit #1)",
        ),
        (
            "PyTorch v2.3 CUDA",
            "PyTorch v2.3 with CUDA",
            "Exact Library Version & Hardware (BM25 & Hybrid hit #1)",
        ),
        (
            "Which database port is configured for Redis?",
            "Redis cache on port 6379",
            "Mixed Semantic + Specific Entity (BM25 & Hybrid hit #1)",
        ),
        (
            "Deep learning framework version update",
            "PyTorch v2.3 with CUDA",
            "Zero Word Overlap Semantic Query (BM25 fails completely; Dense & Hybrid hit #1)",
        ),
    ]

    def evaluate_mode(mode: str) -> tuple[float, float]:
        """Compute MRR and Hit@1 for a given search mode."""
        reciprocal_ranks = []
        hits_at_1 = 0

        for query, target_sub, _ in benchmark_queries:
            results = retrieval.search(user_id, query, top_k=5, mode=mode)
            found_rank = None
            for idx, res in enumerate(results, start=1):
                if target_sub.lower() in res["memory"].content.lower():
                    found_rank = idx
                    break

            if found_rank is not None:
                reciprocal_ranks.append(1.0 / found_rank)
                if found_rank == 1:
                    hits_at_1 += 1
            else:
                reciprocal_ranks.append(0.0)

        mrr = sum(reciprocal_ranks) / len(benchmark_queries)
        hit_1 = hits_at_1 / len(benchmark_queries)
        return mrr, hit_1

    dense_mrr, dense_hit1 = evaluate_mode("dense")
    bm25_mrr, bm25_hit1 = evaluate_mode("sparse")
    hybrid_mrr, hybrid_hit1 = evaluate_mode("hybrid")

    # Output formatted benchmark table
    print("\n" + "=" * 65)
    print("        ADAM RETRIEVAL BENCHMARK: HYBRID vs. DENSE vs. BM25")
    print("=" * 65)
    print(f"{'Search Mode':<18} | {'MRR (Mean Reciprocal Rank)':<28} | {'Hit@1 Accuracy':<15}")
    print("-" * 65)
    print(f"{'Dense Only':<18} | {dense_mrr:<28.4f} | {dense_hit1 * 100:.1f}%")
    print(f"{'BM25 Only':<18} | {bm25_mrr:<28.4f} | {bm25_hit1 * 100:.1f}%")
    print(f"{'Hybrid (RRF)':<18} | {hybrid_mrr:<28.4f} | {hybrid_hit1 * 100:.1f}%")
    print("=" * 65)

    # Hybrid Search must achieve equal or superior performance over both individual baselines
    assert hybrid_mrr >= dense_mrr, f"Hybrid MRR ({hybrid_mrr}) should be >= Dense ({dense_mrr})"
    assert hybrid_mrr >= bm25_mrr, f"Hybrid MRR ({hybrid_mrr}) should be >= BM25 ({bm25_mrr})"
    assert hybrid_hit1 >= dense_hit1, f"Hybrid Hit@1 ({hybrid_hit1}) should be >= Dense ({dense_hit1})"
    assert hybrid_hit1 >= bm25_hit1, f"Hybrid Hit@1 ({hybrid_hit1}) should be >= BM25 ({bm25_hit1})"
    assert hybrid_hit1 == 1.0, f"Hybrid search should achieve 100% Hit@1 on the benchmark suite"
