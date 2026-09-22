"""Hybrid search engine combining dense vector search and BM25 via Reciprocal Rank Fusion (RRF)."""

from typing import Optional, Sequence

from app.memory.models import Memory
from app.retrieval.bm25 import BM25Index
from app.retrieval.similarity import cosine_similarity


def reciprocal_rank_fusion(
    dense_ranked: Sequence[tuple[float, Memory]],
    bm25_ranked: Sequence[tuple[float, Memory]],
    k: int = 60,
) -> list[dict]:
    """Fuse dense vector rankings and BM25 lexical rankings using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(d) = sum( 1 / (k + rank_m(d)) ) for each search method m

    Returns a list of dicts with fused score and per-method diagnostic breakdown.
    """
    scores: dict[str, dict] = {}

    # 1. Process Dense Ranking
    for rank_idx, (score, memory) in enumerate(dense_ranked, start=1):
        mem_id = memory.memory_id
        rrf_contrib = 1.0 / (k + rank_idx) if score > 0.0 else 0.0
        scores[mem_id] = {
            "memory": memory,
            "rrf_score": rrf_contrib,
            "dense_score": float(score),
            "dense_rank": rank_idx if score > 0.0 else None,
            "bm25_score": 0.0,
            "bm25_rank": None,
        }

    # 2. Process BM25 Ranking
    for rank_idx, (score, memory) in enumerate(bm25_ranked, start=1):
        mem_id = memory.memory_id
        rrf_contrib = 1.0 / (k + rank_idx) if score > 0.0 else 0.0
        if mem_id in scores:
            scores[mem_id]["rrf_score"] += rrf_contrib
            scores[mem_id]["bm25_score"] = float(score)
            scores[mem_id]["bm25_rank"] = rank_idx if score > 0.0 else None
        else:
            scores[mem_id] = {
                "memory": memory,
                "rrf_score": rrf_contrib,
                "dense_score": 0.0,
                "dense_rank": None,
                "bm25_score": float(score),
                "bm25_rank": rank_idx if score > 0.0 else None,
            }

    # 3. Sort by total RRF score descending, breaking ties with dense_score, then bm25_score
    ranked = sorted(
        scores.values(),
        key=lambda item: (item["rrf_score"], item["dense_score"], item["bm25_score"]),
        reverse=True,
    )

    # Attach backward-compatible 'similarity' key to each item
    for item in ranked:
        item["similarity"] = round(item["rrf_score"], 6)

    return ranked


class HybridSearchService:
    """Coordinates dense vector search and sparse BM25 retrieval over candidate memories."""

    def __init__(
        self,
        embeddings,
        rrf_k: int = 60,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        self.embeddings = embeddings
        self.rrf_k = rrf_k
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

    def search(
        self,
        query: str,
        memories: list[Memory],
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> list[dict]:
        """Perform search over the provided memories using hybrid, dense, or sparse mode."""
        if not memories:
            return []

        if mode == "dense":
            return self._search_dense(query, memories, top_k)
        elif mode == "sparse":
            return self._search_sparse(query, memories, top_k)
        else:
            return self._search_hybrid(query, memories, top_k)

    def _search_dense(
        self, query: str, memories: list[Memory], top_k: int
    ) -> list[dict]:
        query_embedding = self.embeddings.encode(query)
        ranked = sorted(
            (
                (cosine_similarity(query_embedding, memory.embedding), memory)
                for memory in memories
            ),
            key=lambda item: item[0],
            reverse=True,
        )[:top_k]

        return [
            {
                "memory": memory,
                "similarity": float(score),
                "dense_score": float(score),
                "dense_rank": idx,
                "bm25_score": 0.0,
                "bm25_rank": None,
                "rrf_score": 0.0,
            }
            for idx, (score, memory) in enumerate(ranked, start=1)
        ]

    def _search_sparse(
        self, query: str, memories: list[Memory], top_k: int
    ) -> list[dict]:
        documents = [m.content for m in memories]
        bm25 = BM25Index(documents, k1=self.bm25_k1, b=self.bm25_b)
        ranked_indices = bm25.search(query, top_k=top_k)

        return [
            {
                "memory": memories[doc_idx],
                "similarity": float(score),
                "dense_score": 0.0,
                "dense_rank": None,
                "bm25_score": float(score),
                "bm25_rank": idx,
                "rrf_score": 0.0,
            }
            for idx, (score, doc_idx) in enumerate(ranked_indices, start=1)
        ]

    def _search_hybrid(
        self, query: str, memories: list[Memory], top_k: int
    ) -> list[dict]:
        # 1. Dense retrieval
        query_embedding = self.embeddings.encode(query)
        dense_ranked = sorted(
            (
                (cosine_similarity(query_embedding, memory.embedding), memory)
                for memory in memories
            ),
            key=lambda item: item[0],
            reverse=True,
        )

        # 2. Sparse BM25 retrieval
        documents = [m.content for m in memories]
        bm25 = BM25Index(documents, k1=self.bm25_k1, b=self.bm25_b)
        bm25_ranked_indices = bm25.search(query)
        bm25_ranked = [(score, memories[idx]) for score, idx in bm25_ranked_indices]

        # 3. Fuse with Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(dense_ranked, bm25_ranked, k=self.rrf_k)
        return fused[:top_k]
