import re

from rank_bm25 import BM25Plus

from src.config.settings import settings
from src.retrieval.corpus import CorpusSnapshot, load_snapshot
from src.retrieval.filters import SearchScope, clean_query
from src.retrieval.types import SearchHit


def tokenize(text: str) -> list[str]:
    # Keep identifiers such as max.poll.interval.ms and payments-api intact.
    return re.findall(r"[a-z0-9_]+(?:[.\-][a-z0-9_]+)*", text.lower())


class KeywordRetriever:
    def __init__(self, snapshot: CorpusSnapshot | None = None):
        self.snapshot = snapshot if snapshot is not None else load_snapshot()
        self.tokens = [tokenize(doc.page_content) for doc in self.snapshot.documents]
        if not self.tokens or not any(self.tokens):
            raise ValueError("Corpus contains no searchable words")
        self.token_sets = [set(tokens) for tokens in self.tokens]
        self.bm25 = BM25Plus(self.tokens)

    def search(self, query: str, *, scope: SearchScope, k: int | None = None) -> list[SearchHit]:
        query_tokens = tokenize(clean_query(query))
        limit = settings.keyword_top_k if k is None else k
        if limit < 1:
            raise ValueError("k must be positive")
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        query_terms = set(query_tokens)
        # Fixed-corpus IDF; filtering and lexical-match checks happen BEFORE top-k.
        eligible = [
            i for i, doc in enumerate(self.snapshot.documents)
            if scope.allows(doc.metadata) and query_terms.intersection(self.token_sets[i])
        ]
        eligible.sort(key=lambda i: (-float(scores[i]), self.snapshot.documents[i].metadata["chunk_id"]))
        return [
            SearchHit(self.snapshot.documents[i], {"bm25": float(scores[i])}, {"keyword": rank})
            for rank, i in enumerate(eligible[:limit], start=1)
        ]
