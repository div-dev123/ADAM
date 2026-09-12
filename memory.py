"""Core memory data model shared by storage and retrieval."""

from dataclasses import dataclass, field


@dataclass
class Memory:
    id: str
    content: str
    embedding: list[float]
    timestamp: float
    tier: str = "working"
    importance: float = 0.0
    mem_type: str = "unclassified"
    compressed: bool = False
    access_count: int = 0
    last_accessed: float | None = None
    related_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.timestamp

    def to_document(self):
        return {
            "_id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "timestamp": self.timestamp,
            "tier": self.tier,
            "importance": self.importance,
            "mem_type": self.mem_type,
            "compressed": self.compressed,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "related_ids": self.related_ids,
        }

    @classmethod
    def from_document(cls, document):
        return cls(
            id=str(document["_id"]),
            content=document["content"],
            embedding=list(document["embedding"]),
            timestamp=document["timestamp"],
            tier=document.get("tier", "working"),
            importance=document.get("importance", 0.0),
            mem_type=document.get("mem_type", "unclassified"),
            compressed=document.get("compressed", False),
            access_count=document.get("access_count", 0),
            last_accessed=document.get("last_accessed"),
            related_ids=list(document.get("related_ids", [])),
        )