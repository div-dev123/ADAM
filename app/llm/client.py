"""Structured LLM client interface with an Ollama implementation."""

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Optional, Literal

from pydantic import BaseModel, Field


ConsolidationAction = Literal["NEW", "DUPLICATE", "RELATED", "CONTRADICTORY"]


class ConsolidationDecision(BaseModel):
    action: ConsolidationAction
    merged_content: Optional[str] = None
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
        retrieved_memories: Optional[list[dict]] = None,
        chat_history: Optional[list[dict]] = None,
        formatted_context: Optional[str] = None,
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
            "Respond ONLY with valid JSON with fields: action, merged_content, reason.\n\n"
            "Actions:\n"
            "- RELATED: new memory adds new non-conflicting information that expands a candidate.\n"
            "  merged_content MUST integrate facts from both memories into one coherent sentence.\n"
            "  Example: stored='I code in Python', new='I also use NumPy' → RELATED\n"
            "- NEW: completely different topic, no meaningful overlap with any candidate.\n"
            "  Example: stored='I code in Python', new='I prefer coffee over tea' → NEW\n\n"
            "Important: Only use RELATED if the new memory genuinely adds new facts to a candidate.\n"
            "If uncertain, prefer NEW.\n\n"
            f"Stored memories:\n{candidate_text}\n\n"
            f"New memory: \"{new_content}\"\n\n"
            "Reply ONLY with JSON: {{\"action\": \"RELATED\" or \"NEW\", \"merged_content\": \"...\", \"reason\": \"...\"}}"
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
        retrieved_memories: Optional[list[dict]] = None,
        chat_history: Optional[list[dict]] = None,
        formatted_context: Optional[str] = None,
    ) -> str:
        """Generate a response using Ollama with injected memory context."""
        if formatted_context:
            context_section = formatted_context
        else:
            context_blocks = []
            user_msg_norm = user_message.strip().lower()
            if retrieved_memories:
                for item in retrieved_memories:
                    mem = item.get("memory")
                    sim = item.get("similarity", 0.0)
                    if mem:
                        tier = getattr(mem, "tier", "UNKNOWN")
                        content = getattr(mem, "content", str(mem))
                        imp = getattr(mem, "importance_score", 0.0)
                        # Don't inject redundant memory identical to the active user message
                        if content.strip().lower() == user_msg_norm:
                            continue
                        context_blocks.append(
                            f"• [{tier} | Importance: {imp:.2f} | Sim: {sim:.2f}] {content}"
                        )

            context_text = "\n".join(context_blocks) if context_blocks else "None available."
            context_section = f"=== RETRIEVED RELEVANT MEMORIES ===\n{context_text}\n==================================="

        system_instruction = (
            "You are ADAM Assistant, an intelligent conversational AI equipped with an Adaptive Dynamic Agent Memory (ADAM).\n"
            "Important guidelines:\n"
            "1. NEVER repeat, mirror, or echo the user's message back to them. Always provide an original, helpful assistant response.\n"
            "2. When the user provides a personal update, fact, preference, or contradiction (such as 'Actually, I now use Rust instead of Python'), warmly and conversationally acknowledge the change (e.g. 'Got it! I\\'ve updated your preference to Rust instead of Python.').\n"
            "3. Use the retrieved memories below if relevant to maintain continuity and answer questions about past context.\n"
            "4. Keep answers concise, direct, and conversational.\n\n"
            f"{context_section}\n"
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
                raw = body.get("response", "").strip()
                if raw.lower().startswith("assistant:"):
                    raw = raw[10:].strip()
                norm_resp = raw.lower().strip("\"' .!?")
                norm_user = user_message.lower().strip("\"' .!?")
                if not raw or norm_resp == norm_user:
                    raw = f"Got it! I've noted that: \"{user_message.strip()}\"."
                return raw
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