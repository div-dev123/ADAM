"""Semantic retrieval for Phase 1."""


def semantic_search(store, embedder, query, top_k=5, tiers=None):
    """Embed a query and return the closest stored memories."""
    return store.search(embedder.embed(query), tiers=tiers, top_k=top_k)
