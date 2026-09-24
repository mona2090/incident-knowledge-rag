import math
from functools import lru_cache

from src.config.settings import settings
from src.retrieval.filters import clean_query
from src.retrieval.types import SearchHit


@lru_cache(maxsize=1)
def get_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(settings.reranker_model, device="cpu", max_length=512)


class Reranker:
    def __init__(self, model=None):
        self.model = model

    def rerank(self, query: str, candidates: list[SearchHit], *, top_n: int | None = None) -> list[SearchHit]:
        query = clean_query(query)
        limit = settings.rerank_top_n if top_n is None else top_n
        if limit < 1:
            raise ValueError("top_n must be positive")
        if not candidates:
            return []
        model = self.model if self.model is not None else get_cross_encoder()
        pairs = [(query, hit.document.page_content) for hit in candidates]
        scores = model.predict(pairs, batch_size=8, show_progress_bar=False)
        if len(scores) != len(candidates):
            raise ValueError("Reranker returned an unexpected number of scores")
        ranked = []
        for hit, value in zip(candidates, scores):
            score = float(value)
            if not math.isfinite(score):
                raise ValueError("Reranker returned a non-finite score")
            ranked.append(SearchHit(hit.document, {**hit.scores, "reranker": score}, dict(hit.ranks)))
        ranked.sort(key=lambda hit: (-hit.scores["reranker"], hit.chunk_id))
        return [SearchHit(hit.document, hit.scores, {**hit.ranks, "reranker": rank})
                for rank, hit in enumerate(ranked[:limit], start=1)]
