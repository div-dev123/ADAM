"""Lightweight local sentence embeddings for Phase 1 retrieval."""

import os


class SentenceTransformerEmbedder:
    """Lazy wrapper around the small all-MiniLM-L6-v2 model."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            os.environ.setdefault("USE_TF", "0")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text):
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        vector = self._load_model().encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_many(self, texts):
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Cannot embed an empty text collection")
        vectors = self._load_model().encode(texts, normalize_embeddings=True)
        return vectors.tolist()