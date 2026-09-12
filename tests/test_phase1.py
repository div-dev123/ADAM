import time

from embeddings import SentenceTransformerEmbedder
from memory import Memory
from memory_store import InMemoryMemoryStore
from retrieval import semantic_search


def test_semantic_search_retrieves_relevant_memory():
    embedder = SentenceTransformerEmbedder()
    store = InMemoryMemoryStore()
    now = time.time()

    store.add(Memory(
        id="java-preference",
        content="I prefer Java for solving DSA problems.",
        embedding=embedder.embed("I prefer Java for solving DSA problems."),
        timestamp=now,
    ))
    store.add(Memory(
        id="travel-plan",
        content="I am planning a weekend trip to the coast.",
        embedding=embedder.embed("I am planning a weekend trip to the coast."),
        timestamp=now,
    ))

    results = semantic_search(
        store, embedder, "Which language do I use for algorithms?", top_k=1
    )

    assert results[0][1].id == "java-preference"
    assert results[0][0] > 0.3