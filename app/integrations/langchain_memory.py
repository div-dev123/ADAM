"""LangChain memory adapter for ADAM (Adaptive Dynamic AI Memory).

Provides drop-in memory integration for LangChain chat workflows, chains,
and LCEL pipelines by delegating all retrieval, scoring, tiering, and
consolidation to ADAM's backend engine.
"""

from typing import Any, Dict, List, Optional, Union
import httpx
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)


class ADAMChatMessageHistory(BaseChatMessageHistory):
    """LangChain chat message history backed by ADAM's write and storage engine.

    Stores conversational turns in memory while automatically syncing them
    to ADAM's backend for importance scoring, heuristic noise filtering,
    and consolidation.
    """

    def __init__(
        self,
        user_id: str,
        base_url: str = "http://127.0.0.1:8000",
        client: Optional[httpx.Client] = None,
        auto_sync: bool = True,
    ):
        self.user_id = user_id
        self.base_url = base_url.rstrip("/")
        self._client = client
        self.auto_sync = auto_sync
        self._messages: List[BaseMessage] = []

    @property
    def messages(self) -> List[BaseMessage]:
        return list(self._messages)

    def _get_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(base_url=self.base_url, timeout=30.0)

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history and optionally sync it to ADAM backend."""
        self._messages.append(message)
        if self.auto_sync and message.content:
            self._sync_message_to_adam(message)

    def _sync_message_to_adam(self, message: BaseMessage) -> Optional[Dict[str, Any]]:
        """Push a message into ADAM's write pipeline via POST /memory."""
        content = str(message.content).strip()
        if not content:
            return None

        client = self._get_client()
        try:
            resp = client.post(
                f"{self.base_url}/memory",
                json={"user_id": self.user_id, "content": content},
            )
            if resp.status_code in (200, 201):
                return resp.json()
        except Exception:
            # Graceful degradation if backend is temporarily unreachable
            pass
        return None

    def clear(self) -> None:
        """Clear local in-memory message history."""
        self._messages.clear()


class ADAMLangChainMemory:
    """ADAM memory adapter for LangChain chat workflows and LCEL pipelines.

    Workflow:
    1. Pre-call (load_memory_variables / get_relevant_context):
       Queries ADAM's /retrieve endpoint with query drift detection and
       BM25 + Dense RRF hybrid search. Formats candidate memories as prompt context.
    2. Post-call (save_context):
       Automatically ingests both user and assistant messages into ADAM's
       dual-turn write path, running noise filtering, importance scoring,
       and consolidation without duplicating memory algorithms.
    """

    def __init__(
        self,
        user_id: str,
        base_url: str = "http://127.0.0.1:8000",
        client: Optional[httpx.Client] = None,
        top_k: int = 5,
        memory_key: str = "relevant_memories",
        chat_history_key: str = "history",
        return_messages: bool = False,
        chat_history: Optional[ADAMChatMessageHistory] = None,
    ):
        self.user_id = user_id
        self.base_url = base_url.rstrip("/")
        self._client = client
        self.top_k = top_k
        self.memory_key = memory_key
        self.chat_history_key = chat_history_key
        self.return_messages = return_messages
        self.chat_history = chat_history or ADAMChatMessageHistory(
            user_id=user_id,
            base_url=self.base_url,
            client=self._client,
            auto_sync=False,  # save_context handles dual ingestion
        )

    def _get_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(base_url=self.base_url, timeout=30.0)

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key, self.chat_history_key]

    def format_memories(self, results: List[Dict[str, Any]], drift_info: Optional[Dict[str, Any]] = None) -> str:
        """Format retrieved ADAM memory records into a coherent prompt context string."""
        if not results:
            return "No relevant past memories found in ADAM."

        lines = ["Relevant memories retrieved from ADAM:"]
        if drift_info and "level" in drift_info:
            lines[0] += f" [Drift: {drift_info['level']}, Scope: {', '.join(drift_info.get('scope_tiers', []))}]"

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

    def retrieve_memories(
        self,
        query: str,
        context: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call ADAM's /retrieve API to search relevant memories with drift detection."""
        client = self._get_client()
        k = top_k or self.top_k
        payload: Dict[str, Any] = {
            "user_id": self.user_id,
            "query": query,
            "top_k": k,
        }
        if context:
            payload["context"] = context
        elif self.chat_history.messages:
            payload["context"] = get_buffer_string(self.chat_history.messages[-3:])

        try:
            resp = client.post(f"{self.base_url}/retrieve", json=payload)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass

        return {"query": query, "results": [], "drift": None}

    def get_relevant_context(
        self,
        query: str,
        context: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """Direct helper to retrieve and format memories as prompt context."""
        retrieval_data = self.retrieve_memories(query=query, context=context, top_k=top_k)
        results = retrieval_data.get("results", [])
        drift = retrieval_data.get("drift")
        return self.format_memories(results, drift_info=drift)

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """LangChain standard hook: retrieve memories before LLM invocation."""
        # Extract query from common LangChain input keys
        query = (
            inputs.get("input")
            or inputs.get("question")
            or inputs.get("query")
            or inputs.get("prompt")
            or ""
        )
        if isinstance(query, list):
            query = " ".join(str(m) for m in query)
        query_str = str(query).strip()

        # Retrieve relevant memories from ADAM
        if query_str:
            formatted_memories = self.get_relevant_context(query_str)
        else:
            formatted_memories = "No relevant past memories found in ADAM."

        if self.return_messages:
            memories_val: Union[str, List[BaseMessage]] = [
                SystemMessage(content=formatted_memories)
            ]
            history_val: Union[str, List[BaseMessage]] = self.chat_history.messages
        else:
            memories_val = formatted_memories
            history_val = get_buffer_string(self.chat_history.messages)

        return {
            self.memory_key: memories_val,
            self.chat_history_key: history_val,
        }

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """LangChain standard hook: ingest user & assistant turns after LLM invocation."""
        user_input = (
            inputs.get("input")
            or inputs.get("question")
            or inputs.get("query")
            or inputs.get("prompt")
            or ""
        )
        assistant_output = (
            outputs.get("output")
            or outputs.get("response")
            or outputs.get("text")
            or outputs.get("answer")
            or ""
        )

        user_str = str(user_input).strip()
        assistant_str = str(assistant_output).strip()

        # 1. Update local chat history
        if user_str:
            self.chat_history.add_user_message(user_str)
        if assistant_str:
            self.chat_history.add_ai_message(assistant_str)

        # 2. Ingest both turns into ADAM backend write path
        client = self._get_client()
        for content in (user_str, assistant_str):
            if not content:
                continue
            try:
                client.post(
                    f"{self.base_url}/memory",
                    json={"user_id": self.user_id, "content": content},
                )
            except Exception:
                pass

    def clear(self) -> None:
        """Clear local conversational message history."""
        self.chat_history.clear()
