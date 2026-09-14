"""FastAPI entry point for ADAM Phase 1."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.llm.client import OllamaClient
from app.memory.consolidation import ConsolidationConfig
from app.memory.importance import HeuristicImportanceScorer, ImportanceWeights
from app.memory.storage import SQLiteStorage
from app.memory.tiers import TierAssigner, TierThresholds
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retrieval import RetrievalService


class MemoryCreateRequest(BaseModel):
    user_id: str = Field(min_length=1)
    content: str = Field(min_length=1)


class MemorySearchRequest(BaseModel):
    user_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    top_k: int = Field(default=settings.default_top_k, ge=1, le=settings.max_top_k)


def build_storage() -> SQLiteStorage:
    return SQLiteStorage(settings.database_path)


def build_retrieval_service() -> RetrievalService:
    weights = ImportanceWeights(
        persistent=settings.importance_persistent_weight,
        length=settings.importance_length_weight,
        recurrence=settings.importance_recurrence_weight,
        recency=settings.importance_recency_weight,
    )
    thresholds = TierThresholds(
        archive=settings.archive_threshold,
        working=settings.working_threshold,
        short_term=settings.short_term_threshold,
    )
    return RetrievalService(
        build_storage(),
        EmbeddingService(settings.embedding_model),
        scorer=HeuristicImportanceScorer(weights),
        tier_assigner=TierAssigner(thresholds),
        llm=OllamaClient(settings.ollama_host, settings.ollama_model),
        consolidation_config=ConsolidationConfig(
            candidate_limit=settings.consolidation_candidate_limit,
            min_similarity=settings.consolidation_min_similarity,
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "retrieval"):
        app.state.retrieval = build_retrieval_service()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "phase": 3}


@app.post("/memory", status_code=201)
def store_memory(request: MemoryCreateRequest):
    try:
        memory = app.state.retrieval.store_memory(request.user_id, request.content)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return memory_to_response(memory)


@app.post("/retrieve")
def search_memories(request: MemorySearchRequest):
    try:
        matches = app.state.retrieval.search(
            request.user_id, request.query, request.top_k
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    results = []
    for result in matches:
        results.append({
            "similarity": result["similarity"],
            "memory": memory_to_response(result["memory"]),
        })
    return {
        "query": request.query,
        "results": results,
    }


def memory_to_response(memory):
    return {
        "memory_id": memory.memory_id,
        "user_id": memory.user_id,
        "content": memory.content,
        "created_at": memory.created_at,
        "last_accessed": memory.last_accessed,
        "access_count": memory.access_count,
        "importance_score": memory.importance_score,
        "tier": memory.tier,
    }