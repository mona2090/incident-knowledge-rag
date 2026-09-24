"""Read the exact Step 4 snapshot without importing model/tokenizer libraries."""
import hashlib
import json
from dataclasses import dataclass

from src.config.settings import CHUNKS_FILE, settings
from langchain_core.documents import Document


@dataclass
class CorpusSnapshot:
    namespace: str
    documents: list[Document]


def load_snapshot() -> CorpusSnapshot:
    if not CHUNKS_FILE.exists():
        raise RuntimeError("Missing chunks.json. Complete Step 4 first.")
    raw = CHUNKS_FILE.read_bytes()
    payload = json.loads(raw)
    if (payload["embedding_model"] != settings.embedding_model
            or payload["embedding_dimension"] != settings.embedding_dimension):
        raise ValueError("Embedding settings changed. Rebuild and reindex the corpus.")
    documents = [Document(**row) for row in payload["chunks"]]
    ids = [doc.metadata["chunk_id"] for doc in documents]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Snapshot must have nonempty, unique chunk IDs")
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return CorpusSnapshot(f"{settings.pinecone_namespace}-{digest}", documents)
