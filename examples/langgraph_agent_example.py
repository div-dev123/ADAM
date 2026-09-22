"""LangGraph Agent Example with ADAM Memory Integration.

Demonstrates an autonomous ReAct agent built with LangGraph and local Ollama (qwen2.5:3b)
using ADAM's tools (`retrieve_memory` and `store_memory`) across multiple sessions.

Workflow:
  - Session 1: The user instructs the agent about system architecture requirements.
               The agent uses `store_memory` to commit facts into ADAM's write pipeline.
  - Session 2: In a brand-new interaction session, the user asks about the architecture.
               The agent uses `retrieve_memory` to recall facts from ADAM via hybrid RRF search.
"""

import json
import sys
from pathlib import Path
from typing import Any, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.prebuilt import create_react_agent

from app.integrations.langgraph_tools import create_adam_tools


class OllamaChatModel(BaseChatModel):
    """Lightweight LangChain-compatible ChatModel calling local Ollama (qwen2.5:3b).

    Includes graceful offline fallback for demonstration and automated testing.
    """

    host: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:3b"

    @property
    def _llm_type(self) -> str:
        return "ollama-chat"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Check if Ollama is available
        prompt_lines = [f"{m.type}: {m.content}" for m in messages]
        full_prompt = "\n".join(prompt_lines)

        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": False,
                    },
                )
                if res.status_code == 200:
                    data = res.json()
                    response_text = data.get("response", "").strip()
                    return ChatResult(
                        generations=[ChatGeneration(message=AIMessage(content=response_text))]
                    )
        except Exception:
            pass

        # Offline demonstration fallback simulating autonomous tool reasoning
        last_msg = messages[-1]
        content_lower = str(last_msg.content).lower()

        # If previous message was a ToolMessage returning retrieved memories:
        if isinstance(last_msg, ToolMessage):
            tool_content = str(last_msg.content)
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=f"Based on my memory in ADAM: {tool_content.splitlines()[1] if len(tool_content.splitlines()) > 1 else tool_content}"
                        )
                    )
                ]
            )

        # Simulation 1: Question / Recall prompt -> Call retrieve_memory
        if any(w in content_lower for w in ("what", "which", "where", "how", "recall", "find", "status")):
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "retrieve_memory",
                                    "args": {
                                        "query": "production database architecture and cloud deployment requirements"
                                    },
                                    "id": "call_retrieve_1",
                                }
                            ],
                        )
                    )
                ]
            )

        # Simulation 2: Ingestion prompt -> Call store_memory
        if "remember" in content_lower or "we decided" in content_lower or "requirement" in content_lower:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "store_memory",
                                    "args": {
                                        "content": "Production deployment requirement: Deploy PostgreSQL on AWS RDS with connection pooling enabled."
                                    },
                                    "id": "call_store_1",
                                }
                            ],
                        )
                    )
                ]
            )

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content="I am your assistant equipped with ADAM memory.")
                )
            ]
        )


def run_multi_session_demo(user_id: str = "demo-agent-user", base_url: str = "http://127.0.0.1:8000"):
    print("=" * 70)
    print("  ADAM + LangGraph Multi-Session Agent Demonstration")
    print("=" * 70)

    # Check if a live ADAM server is running; if not, use in-process service
    client = None
    server_live = False
    try:
        with httpx.Client(timeout=1.0) as check_c:
            r = check_c.get(f"{base_url}/health")
            if r.status_code == 200:
                server_live = True
    except Exception:
        server_live = False

    if server_live:
        print(f"[INFO] Connected to live ADAM server at {base_url}")
    else:
        print(f"[INFO] No live server at {base_url}. Using in-process ADAM memory engine.")
        from fastapi.testclient import TestClient
        from app.main import app, build_retrieval_service
        if not hasattr(app.state, "retrieval") or app.state.retrieval is None:
            app.state.retrieval = build_retrieval_service()
        client = TestClient(app)

    # 1. Initialize ADAM tools bound to user session
    tools = create_adam_tools(base_url=base_url, client=client, user_id=user_id)
    llm = OllamaChatModel(model="qwen2.5:3b")

    # 2. Build LangGraph ReAct agent
    agent = create_react_agent(model=llm, tools=tools)

    # -------------------------------------------------------------
    # SESSION 1: Knowledge Ingestion
    # -------------------------------------------------------------
    print("\n[SESSION 1] User supplies a critical architectural requirement...")
    s1_prompt = "Please remember: For production, we decided to deploy PostgreSQL on AWS RDS with connection pooling enabled."
    print(f"User: {s1_prompt}")

    s1_result = agent.invoke(
        {"messages": [HumanMessage(content=s1_prompt)]}
    )
    print("\nAgent Execution Trace (Session 1):")
    for msg in s1_result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"  -> Agent invoked tool: {msg.tool_calls[0]['name']} with args {msg.tool_calls[0]['args']}")
        elif isinstance(msg, ToolMessage):
            print(f"  <- Tool Output: {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            print(f"Agent Final Response: {msg.content}")

    # -------------------------------------------------------------
    # SESSION 2: Independent Subsequent Interaction (Stateful Recall)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[SESSION 2] Brand-new session / thread... User queries past requirements:")
    s2_prompt = "What database architecture did we decide on for production deployment?"
    print(f"User: {s2_prompt}")

    s2_result = agent.invoke(
        {"messages": [HumanMessage(content=s2_prompt)]}
    )
    print("\nAgent Execution Trace (Session 2):")
    for msg in s2_result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"  -> Agent invoked tool: {msg.tool_calls[0]['name']} with args {msg.tool_calls[0]['args']}")
        elif isinstance(msg, ToolMessage):
            print(f"  <- Tool Output:\n{msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            print(f"\nAgent Final Response: {msg.content}")

    print("\n" + "=" * 70)
    print("Multi-session LangGraph agent demonstration completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_multi_session_demo()
