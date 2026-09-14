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


class OllamaClient(LLMClient):
    """Local Ollama client using JSON mode and Pydantic validation."""

    def __init__(self, host: str, model: str, timeout: float = 120.0):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def classify_memory(self, new_content: str, candidates: list[dict]) -> ConsolidationDecision:
        candidate_text = "\n".join(
            f"- {item['memory_id']}: {item['content']}"
            for item in candidates
        ) or "- none"
        prompt = (
            "Classify the new memory against the candidate memories. Return JSON "
            "only with action, merged_content, and reason. action must be exactly "
            "NEW, DUPLICATE, RELATED, or CONTRADICTORY. For RELATED, merge useful "
            "facts. For CONTRADICTORY, keep the newer/current information.\n\n"
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