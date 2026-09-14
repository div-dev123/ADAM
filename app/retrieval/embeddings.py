"""Lightweight local SentenceTransformer embeddings."""

import os


class EmbeddingService:
    def __init__(self, model_name: str, model=None):
        self.model_name = model_name
        self._model = model

    def _load_model(self):
        if self._model is None:
            os.environ.setdefault("USE_TF", "0")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Text cannot be empty")
        vector = self._load_model().encode(text, normalize_embeddings=True)
        return vector.tolist() if hasattr(vector, "tolist") else list(vector)