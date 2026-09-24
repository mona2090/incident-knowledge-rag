from typing import TypedDict

from src.generation.schemas import GenerationResult
from src.retrieval.filters import SearchScope
from src.retrieval.types import SearchHit


class RAGState(TypedDict, total=False):
    query: str
    service: str | None
    environment: str
    technology: str | None
    document_type: str | None
    scope: SearchScope
    dense_hits: list[SearchHit]
    keyword_hits: list[SearchHit]
    candidates: list[SearchHit]
    reranked: list[SearchHit]
    warnings: list[str]
    metrics: dict[str, float | int]
    generation: GenerationResult
    status: str
