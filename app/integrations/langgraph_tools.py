"""LangGraph agent tools for ADAM (Adaptive Dynamic AI Memory).

Exposes ADAM's memory retrieval and storage engine as first-class agent tools
compatible with LangGraph ReAct agents, state graphs, and multi-agent workflows.
"""

from typing import Any, Dict, List, Optional
import httpx
from langchain_core.tools import BaseTool, tool


def format_retrieval_output(retrieval_data: Dict[str, Any]) -> str:
    """Format ADAM /retrieve response into a concise, actionable report for an LLM agent."""
    results = retrieval_data.get("results", [])
    if not results:
        return "No relevant memories found in ADAM memory storage."

    drift = retrieval_data.get("drift")
    lines = ["Relevant memories retrieved from ADAM:"]
    if drift and "level" in drift:
        lines[0] += f" [Drift: {drift['level']}, Scope: {', '.join(drift.get('scope_tiers', []))}]"

    for idx, item in enumerate(results, 1):
        mem = item.get("memory", {})
        content = mem.get("content") or item.get("content", "")
        tier = item.get("memory_tier") or mem.get("tier", "UNKNOWN")
        importance = item.get("importance") or mem.get("importance_score", 0.0)
        similarity = item.get("similarity", 0.0)
        lines.append(
            f"{idx}. [{tier}] {content} (Importance: {importance:.2f}, Similarity: {similarity:.2f})"
        )
    return "\n".join(lines)


def format_store_output(store_data: Dict[str, Any]) -> str:
    """Format ADAM /memory response into a confirmation for an LLM agent."""
    if not store_data.get("is_stored", True) and "message" in store_data:
        return f"ADAM Noise Filter: {store_data['message']}"

    mem_id = store_data.get("memory_id", "created")
    tier = store_data.get("tier", "WORKING")
    importance = store_data.get("importance_score", 0.0)
    return (
        f"Memory successfully stored in ADAM (ID: {mem_id}, Tier: {tier}, "
        f"Importance: {importance:.2f})."
    )


@tool
def retrieve_memory(
    query: str,
    user_id: str,
    top_k: int = 5,
    context: Optional[str] = None,
) -> str:
    """Retrieve relevant memories from ADAM storage for a specific user.

    Uses adaptive query drift detection and hybrid BM25 + Dense RRF vector search
    to find the most relevant past facts, preferences, or decisions.

    Args:
        query: The search question or semantic topic to recall.
        user_id: The unique identifier of the user or session.
        top_k: Maximum number of memory records to return (default 5).
        context: Optional recent conversation context for drift scope selection.
    """
    base_url = "http://127.0.0.1:8000"
    payload: Dict[str, Any] = {
        "user_id": user_id,
        "query": query,
        "top_k": top_k,
    }
    if context:
        payload["context"] = context

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{base_url}/retrieve", json=payload)
            if resp.status_code == 200:
                return format_retrieval_output(resp.json())
            return f"Error retrieving memories: HTTP {resp.status_code} - {resp.text}"
    except Exception as exc:
        return f"Failed to connect to ADAM memory service: {exc}"


@tool
def store_memory(content: str, user_id: str) -> str:
    """Store a new fact, preference, technical decision, or observation into ADAM.

    Routes the input through ADAM's full write pipeline:
    - Heuristic multi-signal importance scoring (0.0 - 1.0)
    - Operational lifecycle tier assignment (WORKING, SHORT_TERM, ARCHIVE)
    - Candidate search and consolidation (NEW, DUPLICATE, RELATED, CONTRADICTORY)
    - Immutable audit trail recording

    Args:
        content: The informative text or factual knowledge to store.
        user_id: The unique identifier of the user or session.
    """
    base_url = "http://127.0.0.1:8000"
    payload = {"user_id": user_id, "content": content}

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{base_url}/memory", json=payload)
            if resp.status_code in (200, 201):
                return format_store_output(resp.json())
            return f"Error storing memory: HTTP {resp.status_code} - {resp.text}"
    except Exception as exc:
        return f"Failed to connect to ADAM memory service: {exc}"


def create_adam_tools(
    base_url: str = "http://127.0.0.1:8000",
    client: Optional[httpx.Client] = None,
    user_id: Optional[str] = None,
) -> List[BaseTool]:
    """Factory creating LangGraph/LangChain tools bound to an ADAM server or specific user session.

    If user_id is provided, the returned tools are pre-bound so the agent does not
    need to pass user_id manually on each tool invocation.
    """
    clean_base_url = base_url.rstrip("/")

    def _execute_retrieve(
        q: str, uid: str, k: int = 5, ctx: Optional[str] = None
    ) -> str:
        payload: Dict[str, Any] = {"user_id": uid, "query": q, "top_k": k}
        if ctx:
            payload["context"] = ctx
        try:
            c = client or httpx.Client(timeout=30.0)
            close_c = client is None
            try:
                resp = c.post(f"{clean_base_url}/retrieve", json=payload)
                if resp.status_code == 200:
                    return format_retrieval_output(resp.json())
                return f"Error retrieving memories: HTTP {resp.status_code} - {resp.text}"
            finally:
                if close_c:
                    c.close()
        except Exception as exc:
            return f"Failed to connect to ADAM memory service: {exc}"

    def _execute_store(c_text: str, uid: str) -> str:
        payload = {"user_id": uid, "content": c_text}
        try:
            c = client or httpx.Client(timeout=30.0)
            close_c = client is None
            try:
                resp = c.post(f"{clean_base_url}/memory", json=payload)
                if resp.status_code in (200, 201):
                    return format_store_output(resp.json())
                return f"Error storing memory: HTTP {resp.status_code} - {resp.text}"
            finally:
                if close_c:
                    c.close()
        except Exception as exc:
            return f"Failed to connect to ADAM memory service: {exc}"

    if user_id:
        # Pre-bound session tools
        @tool
        def session_retrieve_memory(
            query: str, top_k: int = 5, context: Optional[str] = None
        ) -> str:
            """Retrieve relevant memories from ADAM for the current user session."""
            return _execute_retrieve(query, user_id, top_k, context)

        @tool
        def session_store_memory(content: str) -> str:
            """Store a new fact, preference, or decision into ADAM for the current user session."""
            return _execute_store(content, user_id)

        session_retrieve_memory.name = "retrieve_memory"
        session_store_memory.name = "store_memory"
        return [session_retrieve_memory, session_store_memory]

    # Unbound tools
    @tool
    def unbound_retrieve_memory(
        query: str, user_id: str, top_k: int = 5, context: Optional[str] = None
    ) -> str:
        """Retrieve relevant memories from ADAM memory storage for a specified user."""
        return _execute_retrieve(query, user_id, top_k, context)

    @tool
    def unbound_store_memory(content: str, user_id: str) -> str:
        """Store a new fact, preference, or decision into ADAM memory storage for a specified user."""
        return _execute_store(content, user_id)

    unbound_retrieve_memory.name = "retrieve_memory"
    unbound_store_memory.name = "store_memory"
    return [unbound_retrieve_memory, unbound_store_memory]
