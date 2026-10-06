"""Benchmark dataset schemas, generators, and serialization for ADAM experiments."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ConversationTurn:
    turn_id: int
    role: str  # "user" or "assistant"
    content: str
    category: str  # "fact", "filler", "repeat", "contradiction", "preference"
    fact_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class EvalQuery:
    query_id: str
    query: str
    target_facts: List[str] = field(default_factory=list)
    obsolete_facts: List[str] = field(default_factory=list)
    category: str = "general"


@dataclass
class BenchmarkDataset:
    name: str
    description: str
    user_id: str
    turns: List[ConversationTurn]
    queries: List[EvalQuery]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "user_id": self.user_id,
            "turns": [asdict(t) for t in self.turns],
            "queries": [asdict(q) for q in self.queries],
        }

    def save_json(self, path: Union[Path, str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkDataset":
        turns = [ConversationTurn(**t) for t in data["turns"]]
        queries = [EvalQuery(**q) for q in data["queries"]]
        return cls(
            name=data["name"],
            description=data["description"],
            user_id=data.get("user_id", "research_user_1"),
            turns=turns,
            queries=queries,
        )

    @classmethod
    def load_json(cls, path: Union[Path, str]) -> "BenchmarkDataset":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_default_benchmark_dataset() -> BenchmarkDataset:
    """Constructs the canonical 14-turn synthetic research dataset for ADAM evaluation."""
    turns = [
        ConversationTurn(
            turn_id=1,
            role="user",
            content="Hello there!",
            category="filler",
        ),
        ConversationTurn(
            turn_id=2,
            role="user",
            content="I am architecting a high-throughput microservice backend in Rust and PostgreSQL.",
            category="fact",
            fact_id="tech_stack_v1",
            tags=["tech_stack", "backend", "initial"],
        ),
        ConversationTurn(
            turn_id=3,
            role="user",
            content="Thanks!",
            category="filler",
        ),
        ConversationTurn(
            turn_id=4,
            role="user",
            content="I currently live in New York and work remotely from Manhattan.",
            category="fact",
            fact_id="location_v1",
            tags=["location", "initial"],
        ),
        ConversationTurn(
            turn_id=5,
            role="user",
            content="Just a reminder: all of our backend microservices must be built in Rust.",
            category="repeat",
            fact_id="tech_stack_v1_repeat",
            tags=["tech_stack", "repeat"],
        ),
        ConversationTurn(
            turn_id=6,
            role="user",
            content="I enjoy sipping green tea while debugging async code in the afternoon.",
            category="preference",
            fact_id="drink_preference",
            tags=["preference"],
        ),
        ConversationTurn(
            turn_id=7,
            role="user",
            content="Our internal Kafka broker cluster is located at kafka.prod.internal:9092.",
            category="fact",
            fact_id="kafka_broker",
            tags=["infrastructure", "config"],
        ),
        ConversationTurn(
            turn_id=8,
            role="user",
            content="Remember that we use PostgreSQL for all relational data persistence.",
            category="repeat",
            fact_id="database_repeat",
            tags=["database", "repeat"],
        ),
        ConversationTurn(
            turn_id=9,
            role="user",
            content="I moved to Seattle from New York last week, so my primary residence is now Seattle.",
            category="contradiction",
            fact_id="location_v2",
            tags=["location", "update"],
        ),
        ConversationTurn(
            turn_id=10,
            role="user",
            content="We quit Rust and switched to Go because our team hired several Go engineers.",
            category="contradiction",
            fact_id="tech_stack_v2",
            tags=["tech_stack", "update"],
        ),
        ConversationTurn(
            turn_id=11,
            role="user",
            content="Good morning!",
            category="filler",
        ),
        ConversationTurn(
            turn_id=12,
            role="user",
            content="All public API endpoints must maintain latency under 50ms with a p99 SLA.",
            category="fact",
            fact_id="latency_sla",
            tags=["sla", "requirements"],
        ),
        ConversationTurn(
            turn_id=13,
            role="user",
            content="Remember, we now use Go for all new microservice code instead of Rust.",
            category="repeat",
            fact_id="tech_stack_v2_repeat",
            tags=["tech_stack", "update_repeat"],
        ),
        ConversationTurn(
            turn_id=14,
            role="user",
            content="Goodbye!",
            category="filler",
        ),
    ]

    queries = [
        EvalQuery(
            query_id="q1_language",
            query="What programming language does our backend service use?",
            target_facts=["Go", "switched to Go"],
            obsolete_facts=["backend in Rust", "must be built in Rust"],
            category="contradiction_resolution",
        ),
        EvalQuery(
            query_id="q2_location",
            query="Where do I currently live?",
            target_facts=["Seattle", "primary residence is now Seattle"],
            obsolete_facts=["live in New York", "work remotely from Manhattan"],
            category="contradiction_resolution",
        ),
        EvalQuery(
            query_id="q3_kafka",
            query="What is our internal Kafka broker cluster address?",
            target_facts=["kafka.prod.internal:9092"],
            obsolete_facts=[],
            category="factual_recall",
        ),
        EvalQuery(
            query_id="q4_database",
            query="What relational database do we use for data persistence?",
            target_facts=["PostgreSQL"],
            obsolete_facts=[],
            category="factual_recall",
        ),
        EvalQuery(
            query_id="q5_latency",
            query="What are the latency SLA requirements for our API endpoints?",
            target_facts=["latency under 50ms", "p99 SLA"],
            obsolete_facts=[],
            category="factual_recall",
        ),
        EvalQuery(
            query_id="q6_drink",
            query="What beverage do I like while debugging?",
            target_facts=["green tea"],
            obsolete_facts=[],
            category="preference_recall",
        ),
        EvalQuery(
            query_id="q7_out_of_scope",
            query="What is my favorite vacation destination in Europe?",
            target_facts=[],
            obsolete_facts=[],
            category="hallucination_control",
        ),
    ]

    return BenchmarkDataset(
        name="ADAM Canonical Research Benchmark v1",
        description="A controlled multi-turn conversation containing critical facts, filler, repeats, and contradictory updates.",
        user_id="research_eval_user",
        turns=turns,
        queries=queries,
    )
