"""Structured LLM client interface with an Ollama implementation."""

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field


ConsolidationAction = Literal["NEW", "DUPLICATE", "RELATED", "CONTRADICTORY"]


class ConsolidationDecision(BaseModel):
    action: ConsolidationAction
    merged_content: str | None = None
    reason: str = Field(default="", max_length=500)

    def merged_text(self) -> str:
        if self.action in {"RELATED", "CONTRADICTORY"}:
            if not self.merged_content or not self.merged_content.strip():
                raise ValueError(f"{self.action} requires merged_content")
            return self.merged_content.strip()
        return self.merged_content or ""


class CompressionResult(BaseModel):
    compressed_content: str = Field(min_length=1)
    reason: str = Field(default="", max_length=500)


class LLMClient(ABC):
    @abstractmethod
    def classify_memory(self, new_content: str, candidates: list[dict]) -> ConsolidationDecision:
        raise NotImplementedError

    @abstractmethod
    def compress_memory(self, content: str, compression_level: int) -> CompressionResult:
        raise NotImplementedError

    @abstractmethod
    def generate_chat_response(
        self,
        user_message: str,
        retrieved_memories: list[dict] | None = None,
        chat_history: list[dict] | None = None,
    ) -> str:
        raise NotImplementedError

    def check_health(self) -> bool:
        return True


class OllamaClient(LLMClient):
    """Local Ollama client using JSON mode and Pydantic validation."""

    def __init__(self, host: str, model: str, timeout: float = 120.0):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def check_health(self) -> bool:
        """Check if Ollama host is reachable."""
        try:
            request = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(request, timeout=3.0) as response:
                return response.status == 200
        except Exception:
            return False

    def classify_memory(self, new_content: str, candidates: list[dict]) -> ConsolidationDecision:
        candidate_text = "\n".join(
            f"- {item['memory_id']}: {item['content']}"
            for item in candidates
        ) or "- none"
        prompt = (
            "Analyze if the new memory should be consolidated with any candidate memory.\n"
            "Respond in JSON format with fields: action, merged_content, reason.\n"
            "Actions:\n"
            "- RELATED: new memory expands or complements candidate topic; merged_content MUST integrate facts from both.\n"
            "- CONTRADICTORY: new memory conflicts with or supersedes candidate; merged_content contains the updated state.\n"
            "- DUPLICATE: essentially identical memory.\n"
            "- NEW: distinctly different topic/fact, keep separate.\n"
            "If choosing RELATED or CONTRADICTORY, merged_content is required.\n\n"
            f"New memory: {new_content}\nCandidates:\n{candidate_text}"
        )
        return self._request(prompt, ConsolidationDecision)


    def compress_memory(self, content: str, compression_level: int) -> CompressionResult:
        strength = "concise" if compression_level == 1 else "very compact"
        prompt = (
            f"Compress this memory into one {strength} sentence. Preserve factual "
            f"meaning. Return JSON only with compressed_content and reason.\n\n{content}"
        )
        return self._request(prompt, CompressionResult)

    def generate_chat_response(
        self,
        user_message: str,
        retrieved_memories: list[dict] | None = None,
        chat_history: list[dict] | None = None,
    ) -> str:
        """Generate a response using Ollama with injected memory context."""
        context_blocks = []
        if retrieved_memories:
            for item in retrieved_memories:
                mem = item.get("memory")
                sim = item.get("similarity", 0.0)
                if mem:
                    tier = getattr(mem, "tier", "UNKNOWN")
                    content = getattr(mem, "content", str(mem))
                    imp = getattr(mem, "importance_score", 0.0)
                    context_blocks.append(
                        f"• [{tier} | Importance: {imp:.2f} | Sim: {sim:.2f}] {content}"
                    )

        context_text = "\n".join(context_blocks) if context_blocks else "None available."

        system_instruction = (
            "You are ADAM Assistant, an intelligent conversational AI equipped with an Adaptive Memory Management Framework. "
            "Use the retrieved memories below if they are relevant to answer the user accurately, naturally, and concisely.\n\n"
            f"=== RETRIEVED MEMORIES ===\n{context_text}\n"
            "==========================\n"
        )

        history_text = ""
        if chat_history:
            for h in chat_history[-6:]:
                role = "User" if h.get("role") == "user" else "Assistant"
                history_text += f"{role}: {h.get('content', '')}\n"

        prompt = f"{system_instruction}\n{history_text}User: {user_message}\nAssistant:"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3},
        }

        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
                return body.get("response", "").strip()
        except urllib.error.URLError as error:
            raise RuntimeError(f"Could not reach Ollama at {self.host}") from error
        except Exception as error:
            raise RuntimeError(f"Ollama generation failed: {error}") from error

    def _request(self, prompt: str, response_type):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return response_type.model_validate_json(body.get("response", ""))
        except urllib.error.URLError as error:
            raise RuntimeError(f"Could not reach Ollama at {self.host}") from error
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("Ollama returned invalid structured JSON") from error