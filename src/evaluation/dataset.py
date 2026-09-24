import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from src.retrieval.filters import SearchScope


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    query: str = Field(min_length=1, max_length=1000)
    scope: SearchScope
    expected_answerable: bool
    relevant_sources: dict[str, int]
    context_source_ids: list[str]
    reference_answer: str
    tags: list[str] = []

    @model_validator(mode="after")
    def labels_match(self):
        if any(grade not in (1, 2, 3) for grade in self.relevant_sources.values()):
            raise ValueError("Relevance grades must be 1, 2, or 3")
        if self.expected_answerable != bool(self.relevant_sources):
            raise ValueError("Answerable cases need relevance labels; unanswerable cases need none")
        return self


def load_cases(path, snapshot):
    cases = [EvalCase.model_validate(item) for item in json.loads(Path(path).read_text())]
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("Dataset must have unique case IDs and at least one case")
    by_source = {}
    for doc in snapshot.documents:
        by_source.setdefault(doc.metadata["source_id"], []).append(doc)
    for case in cases:
        for source in set(case.relevant_sources) | set(case.context_source_ids):
            if source not in by_source:
                raise ValueError(f"Case {case.id}: source {source} missing from corpus")
            if not any(case.scope.allows(doc.metadata) for doc in by_source[source]):
                raise ValueError(f"Case {case.id}: source {source} violates case filters")
    return cases
