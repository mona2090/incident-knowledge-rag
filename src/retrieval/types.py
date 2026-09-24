from dataclasses import dataclass, field

from langchain_core.documents import Document


@dataclass(frozen=True)
class SearchHit:
    document: Document
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return self.document.metadata["chunk_id"]


@dataclass
class HybridResult:
    candidates: list[SearchHit]
    dense_hits: list[SearchHit]
    keyword_hits: list[SearchHit]
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float | int] = field(default_factory=dict)
