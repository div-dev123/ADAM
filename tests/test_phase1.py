from fastapi.testclient import TestClient

from app.main import app
from app.memory.storage import InMemoryStorage
from app.retrieval.retrieval import RetrievalService


class FakeEmbeddings:
    vectors = {
        "The project is called ADAM and focuses on adaptive memory management.": [1.0, 0.0],
        "ADAM uses semantic memory retrieval to retrieve relevant information.": [0.8, 0.2],
        "What is ADAM?": [0.95, 0.05],
    }

    def encode(self, text):
        return self.vectors[text]


def test_phase1_semantic_retrieval():
    service = RetrievalService(InMemoryStorage(), FakeEmbeddings())
    service.store_memory(
        "user-1",
        "The project is called ADAM and focuses on adaptive memory management.",
    )
    service.store_memory(
        "user-1",
        "ADAM uses semantic memory retrieval to retrieve relevant information.",
    )

    results = service.search("user-1", "What is ADAM?", top_k=2)

    assert len(results) == 2
    assert results[0]["memory"].content.startswith("The project is called ADAM")
    assert results[0]["similarity"] > results[1]["similarity"]


def test_api_health_and_memory_flow():
    app.state.retrieval = RetrievalService(InMemoryStorage(), FakeEmbeddings())
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.json() == {"status": "ok", "phase": 1}

        stored = client.post(
            "/memories",
            json={
                "user_id": "user-1",
                "content": "The project is called ADAM and focuses on adaptive memory management.",
            },
        )
        assert stored.status_code == 201

        searched = client.post(
            "/memories/search",
            json={"user_id": "user-1", "query": "What is ADAM?", "top_k": 1},
        )
        assert searched.status_code == 200
        assert searched.json()["results"][0]["memory"]["content"].startswith(
            "The project is called ADAM"
        )