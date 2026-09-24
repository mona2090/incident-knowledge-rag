import json

from src.config.settings import settings
from src.generation.prompts import SYSTEM_PROMPT
from src.generation.schemas import GenerationResult, GroundedAnswer, abstention
from src.retrieval.filters import clean_query
from src.retrieval.types import SearchHit


def prepare_context(hits: list[SearchHit], *, max_chars: int) -> tuple[str, dict, list[str]]:
    """Keep whole chunks. Character budget is a guard, not a token/cost estimate."""
    records, sources, chunk_ids = [], {}, []
    for hit in hits:
        if hit.chunk_id in chunk_ids:
            continue
        metadata = hit.document.metadata
        citation_id = f"C{len(records) + 1}"
        record = {"citation_id": citation_id, "source_id": metadata["source_id"],
                  "title": metadata["title"], "text": hit.document.page_content}
        proposed = json.dumps(records + [record], ensure_ascii=False)
        if len(proposed) > max_chars:
            continue
        records.append(record)
        chunk_ids.append(hit.chunk_id)
        sources[citation_id] = {
            "source_id": metadata["source_id"], "chunk_id": hit.chunk_id,
            "title": metadata["title"], "source_path": metadata.get("source_path", ""),
        }
    return json.dumps(records, ensure_ascii=False), sources, chunk_ids


def validate_citations(answer: GroundedAnswer, sources: dict) -> None:
    for claim in answer.claims:
        for citation in claim.citations:
            if citation not in sources:
                raise ValueError("Generated citation was not in the supplied context")


class Generator:
    def __init__(self, structured_model=None):
        self.structured_model = structured_model

    def _model(self):
        if self.structured_model is None:
            key = settings.anthropic_api_key.get_secret_value()
            if not key or settings.anthropic_model.startswith("your-"):
                raise ValueError("Set ANTHROPIC_API_KEY and ANTHROPIC_MODEL in .env")
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model=settings.anthropic_model, api_key=key,
                max_tokens=settings.generation_max_tokens,
                timeout=settings.generation_timeout_seconds, max_retries=2,
            )
            # No temperature override: compatibility with models that reject it.
            self.structured_model = llm.with_structured_output(
                GroundedAnswer, method="function_calling", include_raw=True,
            )
        return self.structured_model

    def generate(self, query: str, hits: list[SearchHit]) -> GenerationResult:
        query = clean_query(query)
        context, sources, chunk_ids = prepare_context(hits, max_chars=settings.context_max_chars)
        if not chunk_ids:
            return abstention("No usable evidence fits the selected scope and context budget.")
        payload = json.dumps({"question": query, "evidence": json.loads(context)}, ensure_ascii=False)
        response = self._model().invoke([
            ("system", SYSTEM_PROMPT), ("human", payload),
        ])
        raw = response.get("raw")
        usage = dict(getattr(raw, "usage_metadata", None) or {})
        metadata = getattr(raw, "response_metadata", None) or {}
        parsed = response.get("parsed")
        try:
            if response.get("parsing_error") or parsed is None:
                raise ValueError("Structured output parsing failed")
            if metadata.get("stop_reason") == "max_tokens":
                raise ValueError("Generation reached the output token limit")
            answer = GroundedAnswer.model_validate(parsed)
            validate_citations(answer, sources)
        except ValueError as exc:
            result = abstention("The generated response failed validation; review the evidence directly.")
            result.status = "error"
            result.context_chunk_ids = chunk_ids
            result.usage = usage
            result.warnings = [f"Generation output rejected ({type(exc).__name__})."]
            return result

        cited = {citation for claim in answer.claims for citation in claim.citations}
        return GenerationResult(
            status="answered" if answer.answerable else "abstained", answer=answer,
            sources={key: value for key, value in sources.items() if key in cited},
            context_chunk_ids=chunk_ids, usage=usage,
        )
