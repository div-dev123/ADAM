"""Structured LLM client interface and local Ollama implementation."""

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field


ConsolidationAction = Literal[
    "DUPLICATE", "RELATED", "CONTRADICTORY", "NEW/UNRELATED"
]


class ConsolidationDecision(BaseModel):
    """Validated result returned by a consolidation model."""

    action: ConsolidationAction
    merged_content: str | None = None
    reason: str = Field(default="", max_length=500)

    def require_merged_content(self) -> str:
        if self.action in {"RELATED", "CONTRADICTORY"}:
            if not self.merged_content or not self.merged_content.strip():
                raise ValueError(
                    f"{self.action} decisions require merged_content"
                )
            return self.merged_content.strip()
        return self.merged_content or ""


class LLMClient(ABC):
    """Provider contract used by consolidation logic."""

    @abstractmethod
    def analyze_consolidation(
        self, new_content: str, candidates: list[dict]
    ) -> ConsolidationDecision:
        raise NotImplementedError


class OllamaClient(LLMClient):
    """Call a local Ollama model with JSON-only structured output."""

    def __init__(self, host: str, model: str, timeout: float = 120.0):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def analyze_consolidation(
        self, new_content: str, candidates: list[dict]
    ) -> ConsolidationDecision:
        payload = {
            "model": self.model,
            "prompt": self._build_prompt(new_content, candidates),
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
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}: {error}"
            ) from error
        try:
            decision = ConsolidationDecision.model_validate_json(
                result.get("response", "")
            )
            decision.require_merged_content()
            return decision
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("Ollama returned an invalid consolidation decision") from error

    @staticmethod
    def _build_prompt(new_content: str, candidates: list[dict]) -> str:
        candidate_text = "\n".join(
            f"- id={candidate['memory_id']}: {candidate['content']}"
            for candidate in candidates
        ) or "- No candidate memories"
        return (
            "Classify the new memory against candidate memories. Return JSON "
            "only with keys action, merged_content, and reason. action must be "
            "exactly DUPLICATE, RELATED, CONTRADICTORY, or NEW/UNRELATED. "
            "For RELATED, merge useful information into one concise memory. "
            "For CONTRADICTORY, merged_content must state the newer/current "
            "information. For DUPLICATE and NEW/UNRELATED, merged_content may "
            "be null.\n\n"
            f"New memory: {new_content}\nCandidates:\n{candidate_text}"
        )