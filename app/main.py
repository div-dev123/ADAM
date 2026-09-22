from datetime import timedelta
from typing import Optional
"""FastAPI entry point for ADAM Research Prototype and Web Interface."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app.llm.client import OllamaClient
from app.memory.compression import CompressionConfig
from app.memory.consolidation import ConsolidationConfig
from app.memory.importance import HeuristicImportanceScorer, ImportanceWeights
from app.memory.lifecycle import LifecyclePolicyConfig
from app.memory.storage import SQLiteStorage
from app.memory.tiers import TierAssigner
from app.retrieval.drift import DriftConfig
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retrieval import RetrievalService


class LifecycleRunRequest(BaseModel):
    user_id: Optional[str] = Field(default=None)
    dry_run: bool = Field(default=False)


class MemoryCreateRequest(BaseModel):
    user_id: str = Field(default="user-1", min_length=1)
    content: str = Field(min_length=1)


class MemorySearchRequest(BaseModel):
    user_id: str = Field(default="user-1", min_length=1)
    query: str = Field(min_length=1)
    top_k: int = Field(default=settings.default_top_k, ge=1, le=settings.max_top_k)
    mode: Optional[str] = Field(default=None, pattern="^(hybrid|dense|sparse)$")
    context: Optional[str] = Field(default=None)
    chat_history: list[dict] = Field(default_factory=list)
    scope_tiers: Optional[list[str]] = Field(default=None)


class ChatTurnRequest(BaseModel):
    user_id: str = Field(default="user-1", min_length=1)
    message: str = Field(min_length=1)
    top_k: int = Field(default=settings.default_top_k, ge=1, le=settings.max_top_k)
    chat_history: list[dict] = Field(default_factory=list)


class TransitionRequest(BaseModel):
    target_tier: str = Field(pattern="^(WORKING|SHORT_TERM|LONG_TERM|ARCHIVE)$")


class ResetRequest(BaseModel):
    confirm: bool = Field(default=False)


def build_storage() -> SQLiteStorage:
    return SQLiteStorage(settings.database_path)


def build_retrieval_service() -> RetrievalService:
    weights = ImportanceWeights(
        intent=settings.importance_intent_weight,
        specificity=settings.importance_specificity_weight,
        durability=settings.importance_durability_weight,
        salience=settings.importance_salience_weight,
        recurrence=settings.importance_recurrence_weight,
        recency=settings.importance_recency_weight,
    )
    lifecycle_policy = TierAssigner(
        initial_archive_threshold=settings.initial_archive_threshold,
        initial_long_term_threshold=settings.initial_long_term_threshold,
        working_to_long_term_age=settings.working_to_long_term_age,
        working_to_long_term_min_accesses=settings.working_to_long_term_min_accesses,
        short_term_to_archive_age=settings.short_term_to_archive_age,
        archive_compression_level=settings.archive_compression_level,
    )
    lifecycle_config = LifecyclePolicyConfig(
        forgetting_threshold=settings.lifecycle_forgetting_threshold,
        protected_importance=settings.lifecycle_protected_importance,
        protected_access_count=settings.lifecycle_protected_access_count,
        frequent_access_boost_threshold=settings.lifecycle_frequent_access_boost_threshold,
        working_to_long_term_age=settings.working_to_long_term_age,
        short_term_to_archive_age=settings.short_term_to_archive_age,
        archive_obsolete_age=timedelta(days=settings.lifecycle_archive_obsolete_days),
        recency_decay_lambda=settings.lifecycle_recency_decay_rate,
    )
    drift_config = DriftConfig(
        low_drift_threshold=settings.drift_low_threshold,
        high_drift_threshold=settings.drift_high_threshold,
        time_weight=settings.drift_time_weight,
        time_half_life_hours=settings.drift_time_half_life_hours,
    )
    return RetrievalService(
        build_storage(),
        EmbeddingService(settings.embedding_model),
        scorer=HeuristicImportanceScorer(weights),
        lifecycle_policy=lifecycle_policy,
        llm=OllamaClient(settings.ollama_host, settings.ollama_model, timeout=settings.ollama_timeout),
        consolidation_config=ConsolidationConfig(
            candidate_limit=settings.consolidation_candidate_limit,
            min_similarity=settings.consolidation_min_similarity,
        ),
        compression_config=CompressionConfig(
            working_to_long_term_level=settings.working_compression_level,
            short_term_to_archive_level=settings.archive_compression_level_target,
        ),
        search_mode=settings.search_mode,
        rrf_k=settings.rrf_k,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
        lifecycle_config=lifecycle_config,
        drift_config=drift_config,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "retrieval"):
        app.state.retrieval = build_retrieval_service()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

# Mount static files
static_dir = Path(__file__).parent / "static"
if not static_dir.exists():
    static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def serve_index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(
            str(index_file),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return {"message": f"Welcome to {settings.app_name}. Static UI index not yet generated."}


@app.get("/health")
def health():
    return {"status": "ok", "phase": 3}


@app.get("/system/status")
def system_status():
    """Retrieve runtime status of models, storage, and Ollama."""
    ollama_ok = False
    if hasattr(app.state.retrieval, "llm") and app.state.retrieval.llm:
        ollama_ok = app.state.retrieval.llm.check_health()

    return {
        "app_name": settings.app_name,
        "ollama": {
            "host": settings.ollama_host,
            "model": settings.ollama_model,
            "connected": ollama_ok,
        },
        "embedding_model": settings.embedding_model,
        "database": {
            "path": str(settings.database_path),
            "total_memories": app.state.retrieval.storage.count(),
        },
        "config": {
            "consolidation_min_similarity": settings.consolidation_min_similarity,
            "consolidation_candidate_limit": settings.consolidation_candidate_limit,
            "initial_archive_threshold": settings.initial_archive_threshold,
            "initial_long_term_threshold": settings.initial_long_term_threshold,
            "search_mode": settings.search_mode,
            "rrf_k": settings.rrf_k,
            "bm25_k1": settings.bm25_k1,
            "bm25_b": settings.bm25_b,
            "lifecycle_forgetting_threshold": settings.lifecycle_forgetting_threshold,
            "lifecycle_protected_importance": settings.lifecycle_protected_importance,
            "lifecycle_protected_access_count": settings.lifecycle_protected_access_count,
            "drift_low_threshold": settings.drift_low_threshold,
            "drift_high_threshold": settings.drift_high_threshold,
        },
    }


@app.post("/memory", status_code=201)
def store_memory(request: MemoryCreateRequest):
    try:
        memory = app.state.retrieval.store_memory(request.user_id, request.content)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if memory is None:
        return {
            "message": "Greeting or filler ignored - no memory created",
            "is_stored": False,
            "importance_score": 0.0,
            "tier": None,
        }
    return memory_to_response(memory)


@app.post("/retrieve")
def search_memories(request: MemorySearchRequest):
    try:
        matches = app.state.retrieval.search(
            user_id=request.user_id,
            query=request.query,
            top_k=request.top_k,
            mode=request.mode,
            context=request.context,
            chat_history=request.chat_history,
            scope_tiers=request.scope_tiers,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    results = []
    for result in matches:
        item = {
            "similarity": result["similarity"],
            "memory_tier": result.get("memory_tier", result["memory"].tier),
            "importance": result.get("importance", result["memory"].importance_score),
            "drift_level": result.get("drift_level"),
            "drift_score": result.get("drift_score"),
            "scope_selection_reason": result.get("scope_selection_reason"),
            "memory": memory_to_response(result["memory"]),
        }
        if "dense_score" in result:
            item["dense_score"] = round(result["dense_score"], 4)
        if "bm25_score" in result:
            item["bm25_score"] = round(result["bm25_score"], 4)
        if "rrf_score" in result:
            item["rrf_score"] = round(result["rrf_score"], 6)
        results.append(item)

    drift_data = None
    if hasattr(app.state.retrieval, "last_drift_result") and app.state.retrieval.last_drift_result:
        d = app.state.retrieval.last_drift_result
        drift_data = {
            "level": d.drift_level,
            "score": d.drift_score,
            "scope_tiers": d.scope_tiers,
            "reason": d.reason,
        }

    return {
        "query": request.query,
        "drift": drift_data,
        "results": results,
    }


@app.post("/chat")
def chat_turn(request: ChatTurnRequest):
    """Execute complete conversational turn with memory write, retrieval, generation, and metrics."""
    try:
        turn_result = app.state.retrieval.chat_turn(
            user_id=request.user_id,
            message=request.message,
            top_k=request.top_k,
            chat_history=request.chat_history,
        )
        return turn_result
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Chat turn error: {error}") from error


@app.get("/metrics")
def get_metrics(user_id: Optional[str] = Query(default=None)):
    """Return live research metrics."""
    return app.state.retrieval.storage.get_metrics(user_id=user_id)


@app.get("/memories")
def list_memories(
    user_id: Optional[str] = Query(default=None),
    tier: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
):
    """List memories with optional filters."""
    memories = app.state.retrieval.storage.get_all_memories(
        user_id=user_id, tier=tier, search=search
    )
    return [memory_to_response(m) for m in memories]


@app.get("/memory/{memory_id}")
def get_memory(memory_id: str):
    """Get single memory by id."""
    memory = app.state.retrieval.storage.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory_to_response(memory)


@app.get("/memory/{memory_id}/history")
def get_memory_history(memory_id: str):
    """Get audit history for a single memory."""
    memory = app.state.retrieval.storage.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    history = app.state.retrieval.storage.get_history(memory_id)
    return {
        "memory_id": memory_id,
        "content": memory.content,
        "history": history,
    }


@app.post("/memory/{memory_id}/transition")
def transition_memory_tier(memory_id: str, request: TransitionRequest):
    """Explicitly transition a memory to a target tier using compression."""
    try:
        memory = app.state.retrieval.transition_memory(memory_id, request.target_tier)
        return memory_to_response(memory)
    except KeyError:
        raise HTTPException(status_code=404, detail="Memory not found")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/lifecycle/run")
def run_lifecycle(request: Optional[LifecycleRunRequest] = None):
    """Trigger an automated memory lifecycle evaluation and selective forgetting pass."""
    req = request or LifecycleRunRequest()
    report = app.state.retrieval.run_lifecycle_pass(
        user_id=req.user_id,
        dry_run=req.dry_run,
    )
    return report


@app.get("/lifecycle/policy")
def get_lifecycle_policy():
    """Retrieve active mathematical parameters and thresholds for the memory lifecycle."""
    cfg = app.state.retrieval.lifecycle_manager.config
    return {
        "forgetting_threshold": cfg.forgetting_threshold,
        "protected_importance": cfg.protected_importance,
        "protected_access_count": cfg.protected_access_count,
        "frequent_access_boost_threshold": cfg.frequent_access_boost_threshold,
        "working_to_long_term_age_days": cfg.working_to_long_term_age.total_seconds() / 86400.0,
        "working_to_long_term_importance_min": cfg.working_to_long_term_importance_min,
        "short_term_to_archive_age_days": cfg.short_term_to_archive_age.total_seconds() / 86400.0,
        "archive_obsolete_age_days": cfg.archive_obsolete_age.total_seconds() / 86400.0,
        "recency_decay_lambda": cfg.recency_decay_lambda,
        "weight_importance": cfg.weight_importance,
        "weight_recency": cfg.weight_recency,
        "weight_frequency": cfg.weight_frequency,
        "compression_penalty_per_level": cfg.compression_penalty_per_level,
    }


@app.delete("/memory/{memory_id}")
def delete_memory(memory_id: str):
    """Delete a memory."""
    deleted = app.state.retrieval.storage.delete_memory(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "deleted", "memory_id": memory_id}


@app.get("/history")
def list_history(
    limit: int = Query(default=50, ge=1, le=200),
    user_id: Optional[str] = Query(default=None),
):
    """List recent global consolidation and compression history records."""
    return app.state.retrieval.storage.get_all_history(limit=limit, user_id=user_id)


@app.post("/reset")
def reset_database(request: ResetRequest):
    """Reset the research database after explicit confirmation."""
    if not request.confirm:
        raise HTTPException(status_code=400, detail="Confirmation required (confirm: true)")
    app.state.retrieval.storage.reset_database()
    return {"status": "reset", "message": "Research database cleared successfully."}


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
        "compression_level": memory.compression_level,
        "updated_at": memory.updated_at,
        "superseded_by": memory.superseded_by,
        "source_role": memory.source_role,
    }