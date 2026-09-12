"""Memory persistence and cosine-similarity retrieval for ADAM Phase 1."""

import math
import os
import uuid

from memory import Memory


class InMemoryMemoryStore:
    """Small local store used by tests and offline experiments."""

    def __init__(self):
        self._memories = {}

    def add(self, memory):
        self._memories[memory.id] = memory

    def update(self, memory):
        self._memories[memory.id] = memory

    def delete(self, memory_id):
        self._memories.pop(memory_id, None)

    def all(self, tier=None):
        memories = list(self._memories.values())
        return [memory for memory in memories if tier is None or memory.tier == tier]

    def count(self, tier=None):
        return len(self.all(tier))

    def total_chars(self, tier=None):
        return sum(len(memory.content) for memory in self.all(tier))

    def search(self, query_embedding, tiers=None, top_k=10):
        candidates = self.all() if tiers is None else [
            memory for tier in tiers for memory in self.all(tier)
        ]
        scored = [
            (self.cosine(query_embedding, memory.embedding), memory)
            for memory in candidates
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    @staticmethod
    def cosine(a, b):
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1e-9
        norm_b = math.sqrt(sum(y * y for y in b)) or 1e-9
        return dot / (norm_a * norm_b)

    @staticmethod
    def new_id():
        return str(uuid.uuid4())


class MongoMemoryStore:
    """MongoDB Atlas store with Python-side similarity ranking."""

    def __init__(self, uri, database="adam_memory", collection="memories", client=None):
        from pymongo import MongoClient

        self.client = client or MongoClient(uri)
        self.collection = self.client[database][collection]
        self.collection.create_index("tier")

    def add(self, memory):
        self.collection.insert_one(memory.to_document())

    def update(self, memory):
        document = memory.to_document()
        document.pop("_id")
        self.collection.update_one({"_id": memory.id}, {"$set": document})

    def delete(self, memory_id):
        self.collection.delete_one({"_id": memory_id})

    def all(self, tier=None):
        query = {} if tier is None else {"tier": tier}
        return [Memory.from_document(document) for document in self.collection.find(query)]

    def count(self, tier=None):
        query = {} if tier is None else {"tier": tier}
        return self.collection.count_documents(query)

    def total_chars(self, tier=None):
        return sum(len(memory.content) for memory in self.all(tier))

    def search(self, query_embedding, tiers=None, top_k=10):
        candidates = self.all() if tiers is None else [
            memory for tier in tiers for memory in self.all(tier)
        ]
        scored = [
            (InMemoryMemoryStore.cosine(query_embedding, memory.embedding), memory)
            for memory in candidates
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    @staticmethod
    def cosine(a, b):
        return InMemoryMemoryStore.cosine(a, b)

    @staticmethod
    def new_id():
        return InMemoryMemoryStore.new_id()


class MemoryStore:
    """Select Atlas when MONGODB_URI is configured, otherwise stay local."""

    def __init__(self, mongo_uri=None, database="adam_memory", collection="memories"):
        uri = mongo_uri or os.getenv("MONGODB_URI")
        self.backend = (
            MongoMemoryStore(uri, database, collection)
            if uri else InMemoryMemoryStore()
        )

    def __getattr__(self, name):
        return getattr(self.backend, name)

    @staticmethod
    def cosine(a, b):
        return InMemoryMemoryStore.cosine(a, b)

    @staticmethod
    def new_id():
        return InMemoryMemoryStore.new_id()


MemoryUnit = Memory
