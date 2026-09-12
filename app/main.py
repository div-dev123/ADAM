"""FastAPI entry point for ADAM Phase 1."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.memory.storage import InMemoryStorage, MemoryStorage, MongoStorage
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retrieval import RetrievalService


class MemoryCreateRequest(BaseModel):
    user_id: str = Field(min_length=1)
    content: str = Field(min_length=1)


class MemorySearchRequest(BaseModel):
    user_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    top_k: int = Field(default=settings.default_top_k, ge=1, le=settings.max_top_k)


def build_storage() -> MemoryStorage:
    if settings.mongo_uri:
        return MongoStorage(
            settings.mongo_uri,
            settings.mongo_database,
            settings.mongo_collection,
        )
    return InMemoryStorage()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "retrieval"):
        app.state.retrieval = RetrievalService(
            build_storage(), EmbeddingService(settings.embedding_model)
        )
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "phase": 1}


@app.post("/memories", status_code=201)
def store_memory(request: MemoryCreateRequest):
    try:
        memory = app.state.retrieval.store_memory(request.user_id, request.content)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return memory_to_response(memory)


@app.post("/memories/search")
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
    }