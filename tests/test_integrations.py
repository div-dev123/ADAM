"""Unit and integration tests for LangChain and LangGraph adapters in ADAM."""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.prebuilt import create_react_agent

from app.integrations.langchain_memory import (
    ADAMChatMessageHistory,
    ADAMLangChainMemory,
)
from app.integrations.langgraph_tools import (
    create_adam_tools,
    format_retrieval_output,
    format_store_output,
    retrieve_memory,
    store_memory,
)
from app.main import app


# ---------------------------------------------------------------------------
# 1. LangChain Memory Adapter Tests
# ---------------------------------------------------------------------------


def test_adam_chat_message_history_adds_messages():
    """Verify ADAMChatMessageHistory stores messages and formats buffer."""
    history = ADAMChatMessageHistory(user_id="test-user-1", auto_sync=False)
    history.add_user_message("I am building an agent with FastAPI.")
    history.add_ai_message("FastAPI is a great choice for asynchronous APIs.")

    assert len(history.messages) == 2
    assert isinstance(history.messages[0], HumanMessage)
    assert history.messages[0].content == "I am building an agent with FastAPI."
    assert isinstance(history.messages[1], AIMessage)
    assert history.messages[1].content == "FastAPI is a great choice for asynchronous APIs."

    history.clear()
    assert len(history.messages) == 0


def test_adam_chat_message_history_auto_sync():
    """Verify auto_sync calls the backend memory endpoint."""
    mock_client = MagicMock()
    mock_client.post.return_value.status_code = 201
    mock_client.post.return_value.json.return_value = {
        "memory_id": "mem-123",
        "tier": "WORKING",
        "importance_score": 0.85,
    }

    history = ADAMChatMessageHistory(
        user_id="test-user-2", client=mock_client, auto_sync=True
    )
    history.add_user_message("We chose SQLite for local persistence.")

    assert mock_client.post.called
    args, kwargs = mock_client.post.call_args
    assert "memory" in args[0]
    assert kwargs["json"] == {
        "user_id": "test-user-2",
        "content": "We chose SQLite for local persistence.",
    }


def test_adam_langchain_memory_load_memory_variables():
    """Verify load_memory_variables queries ADAM /retrieve and formats context."""
    mock_client = MagicMock()
    mock_client.post.return_value.status_code = 200
    mock_client.post.return_value.json.return_value = {
        "query": "What database are we using?",
        "drift": {
            "level": "LOW",
            "score": 0.25,
            "scope_tiers": ["WORKING", "SHORT_TERM"],
        },
        "results": [
            {
                "similarity": 0.88,
                "memory_tier": "WORKING",
                "importance": 0.85,
                "memory": {
                    "memory_id": "mem-1",
                    "content": "Architecture decision: We chose SQLite for local persistence.",
                    "tier": "WORKING",
                    "importance_score": 0.85,
                },
            }
        ],
    }

    memory = ADAMLangChainMemory(
        user_id="test-user-3",
        client=mock_client,
        memory_key="relevant_memories",
        chat_history_key="history",
    )

    vars_dict = memory.load_memory_variables({"input": "What database are we using?"})

    assert "relevant_memories" in vars_dict
    assert "history" in vars_dict
    context = vars_dict["relevant_memories"]
    assert "Relevant memories retrieved from ADAM:" in context
    assert "[Drift: LOW" in context
    assert "Architecture decision: We chose SQLite" in context
    assert "(Importance: 0.85, Similarity: 0.88)" in context


def test_adam_langchain_memory_save_context():
    """Verify save_context pushes user and assistant turns to ADAM."""
    mock_client = MagicMock()
    mock_client.post.return_value.status_code = 201
    mock_client.post.return_value.json.return_value = {"status": "ok"}

    memory = ADAMLangChainMemory(user_id="test-user-4", client=mock_client)

    inputs = {"input": "We are using SentenceTransformers for embeddings."}
    outputs = {"output": "SentenceTransformers works well with all-MiniLM-L6-v2."}

    memory.save_context(inputs, outputs)

    # Local chat history should have both messages
    assert len(memory.chat_history.messages) == 2
    assert memory.chat_history.messages[0].content == inputs["input"]
    assert memory.chat_history.messages[1].content == outputs["output"]

    # Both turns must have been posted to backend
    assert mock_client.post.call_count == 2
    calls = mock_client.post.call_args_list
    assert calls[0][1]["json"]["content"] == inputs["input"]
    assert calls[1][1]["json"]["content"] == outputs["output"]


def test_adam_langchain_memory_return_messages():
    """Verify return_messages=True formats context and history as BaseMessage objects."""
    mock_client = MagicMock()
    mock_client.post.return_value.status_code = 200
    mock_client.post.return_value.json.return_value = {
        "query": "fastapi",
        "results": [],
        "drift": None,
    }

    memory = ADAMLangChainMemory(
        user_id="test-user-5", client=mock_client, return_messages=True
    )
    memory.chat_history.add_user_message("Hello")

    vars_dict = memory.load_memory_variables({"input": "fastapi"})
    assert isinstance(vars_dict["relevant_memories"], list)
    assert len(vars_dict["relevant_memories"]) == 1
    assert isinstance(vars_dict["history"], list)
    assert len(vars_dict["history"]) == 1


# ---------------------------------------------------------------------------
# 2. LangGraph Agent Tools Tests
# ---------------------------------------------------------------------------


def test_retrieve_memory_tool_schema():
    """Verify retrieve_memory tool exposes correct schema and metadata."""
    assert retrieve_memory.name == "retrieve_memory"
    assert "query" in retrieve_memory.args
    assert "user_id" in retrieve_memory.args
    assert "top_k" in retrieve_memory.args
    assert "context" in retrieve_memory.args


def test_store_memory_tool_schema():
    """Verify store_memory tool exposes correct schema and metadata."""
    assert store_memory.name == "store_memory"
    assert "content" in store_memory.args
    assert "user_id" in store_memory.args


def test_format_retrieval_output():
    """Verify output formatting handles empty results and enriched hits."""
    empty_res = format_retrieval_output({"results": []})
    assert "No relevant memories found" in empty_res

    res = format_retrieval_output(
        {
            "drift": {"level": "MEDIUM", "scope_tiers": ["WORKING", "LONG_TERM"]},
            "results": [
                {
                    "similarity": 0.75,
                    "memory_tier": "LONG_TERM",
                    "importance": 0.80,
                    "memory": {"content": "Use Redis for rate limiting."},
                }
            ],
        }
    )
    assert "[Drift: MEDIUM" in res
    assert "[LONG_TERM] Use Redis for rate limiting." in res
    assert "(Importance: 0.80, Similarity: 0.75)" in res


def test_format_store_output():
    """Verify output formatting handles successful storage and noise filtering."""
    store_ok = format_store_output(
        {"memory_id": "mem-abc", "tier": "WORKING", "importance_score": 0.88}
    )
    assert "Memory successfully stored in ADAM" in store_ok
    assert "mem-abc" in store_ok
    assert "WORKING" in store_ok

    store_noise = format_store_output(
        {
            "is_stored": False,
            "message": "Greeting or filler ignored - no memory created",
        }
    )
    assert "ADAM Noise Filter" in store_noise


def test_create_adam_tools_with_user_binding():
    """Verify create_adam_tools creates pre-bound tools with session user_id."""
    mock_client = MagicMock()
    mock_client.post.return_value.status_code = 200
    mock_client.post.return_value.json.return_value = {
        "results": [
            {
                "similarity": 0.90,
                "memory_tier": "WORKING",
                "importance": 0.85,
                "memory": {"content": "Pre-bound test memory."},
            }
        ]
    }

    tools = create_adam_tools(client=mock_client, user_id="session-user-99")
    assert len(tools) == 2
    tool_names = [t.name for t in tools]
    assert "retrieve_memory" in tool_names
    assert "store_memory" in tool_names

    # Invoking retrieve_memory should NOT require user_id argument
    retrieve_t = next(t for t in tools if t.name == "retrieve_memory")
    res = retrieve_t.invoke({"query": "test query"})
    assert "Pre-bound test memory" in res

    # Verify user_id was automatically injected
    args, kwargs = mock_client.post.call_args
    assert kwargs["json"]["user_id"] == "session-user-99"


# ---------------------------------------------------------------------------
# 3. LangGraph Agent Multi-Turn Simulation Test
# ---------------------------------------------------------------------------


class MockAgentChatModel(BaseChatModel):
    """Deterministic mock ChatModel simulating a LangGraph agent calling ADAM tools."""

    turn: int = 0

    @property
    def _llm_type(self) -> str:
        return "mock-agent"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_msg = messages[-1]

        # If previous message is ToolMessage (result from tool), return final response
        if isinstance(last_msg, ToolMessage):
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=f"Agent processed memory: {last_msg.content}"
                        )
                    )
                ]
            )

        # First turn: call retrieve_memory
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "retrieve_memory",
                                "args": {"query": "database choice"},
                                "id": "call_1",
                            }
                        ],
                    )
                )
            ]
        )


def test_langgraph_react_agent_with_adam_tools():
    """Verify a LangGraph create_react_agent successfully invokes ADAM tools."""
    mock_client = MagicMock()
    mock_client.post.return_value.status_code = 200
    mock_client.post.return_value.json.return_value = {
        "results": [
            {
                "similarity": 0.92,
                "memory_tier": "WORKING",
                "importance": 0.88,
                "memory": {"content": "Database selected is PostgreSQL."},
            }
        ]
    }

    tools = create_adam_tools(client=mock_client, user_id="agent-user-1")
    model = MockAgentChatModel()
    agent = create_react_agent(model=model, tools=tools)

    result = agent.invoke({"messages": [HumanMessage(content="Which database?")]})

    messages = result["messages"]
    assert len(messages) >= 3

    # Check that tool call occurred
    ai_tool_msg = messages[1]
    assert hasattr(ai_tool_msg, "tool_calls")
    assert ai_tool_msg.tool_calls[0]["name"] == "retrieve_memory"

    # Check tool response
    tool_resp_msg = messages[2]
    assert isinstance(tool_resp_msg, ToolMessage)
    assert "Database selected is PostgreSQL" in tool_resp_msg.content

    # Check final AI message
    final_ai_msg = messages[3]
    assert isinstance(final_ai_msg, AIMessage)
    assert "Agent processed memory" in final_ai_msg.content
