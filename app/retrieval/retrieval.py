"""Phase 1 semantic retrieval service."""

from app.memory.models import Memory, utc_now
from app.memory.storage import MemoryStorage
from app.retrieval.embeddings import EmbeddingService


class RetrievalService:
    def __init__(self, storage: MemoryStorage, embeddings: EmbeddingService):
        self.storage = storage
        self.embeddings = embeddings

    def store_memory(self, user_id: str, content: str) -> Memory:
        from app.memory.storage import create_memory

        memory = create_memory(user_id, content, self.embeddings.encode(content))
        return self.storage.add(memory)

    def search(self, user_id: str, query: str, top_k: int):
        query_embedding = self.embeddings.encode(query)
        ranked = self.storage.search(user_id, query_embedding, top_k)
        results = []
        for score, memory in ranked:
            memory.last_accessed = utc_now()
            memory.access_count += 1
            if hasattr(self.storage, "update"):
                self.storage.update(memory)
            results.append({"memory": memory, "similarity": score})
        return results