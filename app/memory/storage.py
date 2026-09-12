"""Memory persistence backends for Phase 1."""

import math
import uuid
from abc import ABC, abstractmethod

from app.memory.models import Memory, utc_now


class MemoryStorage(ABC):
    @abstractmethod
    def add(self, memory: Memory) -> Memory:
        raise NotImplementedError

    @abstractmethod
    def update(self, memory: Memory) -> Memory:
        raise NotImplementedError

    @abstractmethod
    def search(self, user_id: str, query_embedding: list[float], top_k: int):
        raise NotImplementedError


class InMemoryStorage(MemoryStorage):
    """Deterministic local backend for tests and development."""

    def __init__(self):
        self._memories: dict[str, Memory] = {}

    def add(self, memory: Memory) -> Memory:
        self._memories[memory.memory_id] = memory
        return memory

    def update(self, memory: Memory) -> Memory:
        self._memories[memory.memory_id] = memory
        return memory

    def search(self, user_id: str, query_embedding: list[float], top_k: int):
        candidates = [
            memory for memory in self._memories.values()
            if memory.user_id == user_id
        ]
        ranked = [
            (cosine_similarity(query_embedding, memory.embedding), memory)
            for memory in candidates
        ]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[:top_k]


class MongoStorage(MemoryStorage):
    """MongoDB Atlas backend with transparent Python similarity ranking."""

    def __init__(self, uri: str, database: str, collection: str):
        from pymongo import MongoClient

        self.client = MongoClient(uri)
        self.collection = self.client[database][collection]
        self.collection.create_index("user_id")

    def add(self, memory: Memory) -> Memory:
        self.collection.insert_one(memory.to_document())
        return memory

    def update(self, memory: Memory) -> Memory:
        document = memory.to_document()
        document.pop("_id")
        self.collection.update_one({"_id": memory.memory_id}, {"$set": document})
        return memory

    def search(self, user_id: str, query_embedding: list[float], top_k: int):
        candidates = [
            Memory.from_document(document)
            for document in self.collection.find({"user_id": user_id})
        ]
        ranked = [
            (cosine_similarity(query_embedding, memory.embedding), memory)
            for memory in candidates
        ]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[:top_k]


def cosine_similarity(first: list[float], second: list[float]) -> float:
    if not first or not second or len(first) != len(second):
        return 0.0
    dot_product = sum(left * right for left, right in zip(first, second))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if not first_norm or not second_norm:
        return 0.0
    return dot_product / (first_norm * second_norm)


def create_memory(user_id: str, content: str, embedding: list[float]) -> Memory:
    now = utc_now()
    return Memory(
        memory_id=str(uuid.uuid4()),
        user_id=user_id,
        content=content,
        embedding=embedding,
        created_at=now,
        last_accessed=now,
    )