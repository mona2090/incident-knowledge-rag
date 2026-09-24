from time import perf_counter

from src.config.settings import settings
from src.retrieval.corpus import CorpusSnapshot, load_snapshot
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.filters import SearchScope, clean_query
from src.retrieval.keyword_retriever import KeywordRetriever
from src.retrieval.types import HybridResult, SearchHit


def reciprocal_rank_fusion(rankings: dict[str, list[SearchHit]], *, rrf_k: int = 60,
                           top_k: int = 10) -> list[SearchHit]:
    if rrf_k < 1 or top_k < 1:
        raise ValueError("RRF parameters must be positive")
    combined = {}
    for branch, hits in rankings.items():
        seen = set()
        for hit in hits:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            rank = len(seen)  # one-based rank over unique chunks
            old = combined.get(hit.chunk_id)
            scores = {**(old.scores if old else {}), **hit.scores}
            scores["rrf"] = (old.scores["rrf"] if old else 0.0) + 1.0 / (rrf_k + rank)
            ranks = {**(old.ranks if old else {}), branch: rank}
            combined[hit.chunk_id] = SearchHit(hit.document, scores, ranks)
    return sorted(combined.values(), key=lambda hit: (-hit.scores["rrf"], hit.chunk_id))[:top_k]


class HybridRetriever:
    def __init__(self, snapshot: CorpusSnapshot | None = None, dense=None, keyword=None):
        self.snapshot = snapshot if snapshot is not None else load_snapshot()
        self.by_id = {doc.metadata["chunk_id"]: doc for doc in self.snapshot.documents}
        self.dense = dense if dense is not None else DenseRetriever(self.snapshot)
        self.keyword = keyword if keyword is not None else KeywordRetriever(self.snapshot)

    def search(self, query: str, *, scope: SearchScope) -> HybridResult:
        query = clean_query(query)
        if not any(scope.allows(doc.metadata) for doc in self.snapshot.documents):
            return HybridResult([], [], [], metrics={"eligible_chunks": 0})

        warnings = []
        metrics = {}
        started = perf_counter()
        try:
            raw_dense = self.dense.search(query, **scope.model_dump(), k=settings.dense_top_k)
        except Exception as exc:
            # Operational fallback is explicit; the error is never reported as success.
            warnings.append(f"Dense retrieval unavailable ({type(exc).__name__}); using BM25 only.")
            raw_dense = []
        metrics["dense_ms"] = round((perf_counter() - started) * 1000, 2)

        dense_hits = []
        seen = set()
        for doc, score in raw_dense:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id not in self.by_id or not scope.allows(doc.metadata):
                raise RuntimeError("Dense result does not match the requested snapshot/scope")
            canonical = self.by_id[chunk_id]
            if not scope.allows(canonical.metadata):
                raise RuntimeError("Local snapshot violated metadata scope")
            if chunk_id not in seen:
                seen.add(chunk_id)
                dense_hits.append(SearchHit(canonical, {"dense": float(score)}, {"dense": len(seen)}))

        started = perf_counter()
        keyword_hits = self.keyword.search(query, scope=scope)
        metrics["keyword_ms"] = round((perf_counter() - started) * 1000, 2)
        started = perf_counter()
        candidates = reciprocal_rank_fusion(
            {"dense": dense_hits, "keyword": keyword_hits},
            rrf_k=settings.rrf_k, top_k=settings.hybrid_top_k,
        )
        metrics.update(fusion_ms=round((perf_counter() - started) * 1000, 2),
                       dense_count=len(dense_hits), keyword_count=len(keyword_hits),
                       candidate_count=len(candidates))
        return HybridResult(candidates, dense_hits, keyword_hits, warnings, metrics)
