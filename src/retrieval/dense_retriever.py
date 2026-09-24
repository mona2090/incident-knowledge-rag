"""Step 5-compatible public API, now using the common metadata policy."""
import argparse

from src.config.settings import settings
from src.retrieval.corpus import CorpusSnapshot, load_snapshot
from src.retrieval.filters import SearchScope, clean_query


class DenseRetriever:
    def __init__(self, snapshot: CorpusSnapshot | None = None, store=None):
        self.snapshot = snapshot if snapshot is not None else load_snapshot()
        self.store = store

    def _store(self):
        if self.store is None:
            # Lazy imports: keyword-only search and offline tests need no ML model.
            from langchain_pinecone import PineconeVectorStore
            from src.ingestion.indexer import connect_index
            from src.retrieval.embeddings import get_embeddings
            self.store = PineconeVectorStore(
                index=connect_index(), embedding=get_embeddings(),
                namespace=self.snapshot.namespace,
            )
        return self.store

    def search(self, query: str, *, service: str | None = None,
               environment: str = "production", technology: str | None = None,
               document_type: str | None = None, k: int | None = None):
        query = clean_query(query)
        limit = settings.dense_top_k if k is None else k
        if not 1 <= limit <= 100:
            raise ValueError("k must be between 1 and 100")
        scope = SearchScope(service=service, environment=environment,
                            technology=technology, document_type=document_type)
        results = self._store().similarity_search_with_score(
            query=query, k=limit, filter=scope.pinecone_filter(),
        )
        if any(not scope.allows(doc.metadata) for doc, _ in results):
            raise RuntimeError("Dense result violated metadata scope")
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--service")
    parser.add_argument("--environment", default="production")
    args = parser.parse_args()
    for doc, score in DenseRetriever().search(args.query, service=args.service, environment=args.environment):
        print(doc.metadata["source_id"], f"cosine={score:.4f}", doc.page_content)
