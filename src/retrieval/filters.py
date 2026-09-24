from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SearchScope(BaseModel):
    """Relevance filters. User authorization must be enforced separately."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    service: str | None = Field(default=None, min_length=1)
    environment: Literal["production", "staging", "development"] = "production"
    technology: str | None = Field(default=None, min_length=1)
    document_type: Literal["incident", "runbook"] | None = None

    def equalities(self) -> dict[str, str]:
        return {"approval_status": "approved", **self.model_dump(exclude_none=True)}

    def allows(self, metadata: dict) -> bool:
        return all(metadata.get(key) == value for key, value in self.equalities().items())

    def pinecone_filter(self) -> dict:
        return {"$and": [{key: {"$eq": value}} for key, value in self.equalities().items()]}


def clean_query(query: str) -> str:
    query = query.strip()
    if not query or len(query) > 1000:
        raise ValueError("Query must contain 1–1000 characters")
    return query
