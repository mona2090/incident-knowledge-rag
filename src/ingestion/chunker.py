import hashlib
import json

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from transformers import AutoTokenizer

from src.config.settings import CHUNKS_FILE, settings
from src.ingestion.loader import load_documents


def load_snapshot() -> tuple[str, list[Document]]:
    if not CHUNKS_FILE.exists():
        raise RuntimeError(
            "Run python -m src.ingestion.chunker first"
        )

    raw = CHUNKS_FILE.read_bytes()
    payload = json.loads(raw)

    if (
        payload["embedding_model"]
        != settings.embedding_model
        or payload["embedding_dimension"]
        != settings.embedding_dimension
    ):
        raise ValueError(
            "Embedding settings changed; rebuild and reindex"
        )

    snapshot_id = hashlib.sha256(raw).hexdigest()[:16]
    namespace = (
        f"{settings.pinecone_namespace}-{snapshot_id}"
    )

    documents = [
        Document(**row)
        for row in payload["chunks"]
    ]

    return namespace, documents


def build_snapshot() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        settings.embedding_model
    )

    splitter = (
        RecursiveCharacterTextSplitter
        .from_huggingface_tokenizer(
            tokenizer,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    )

    sources = load_documents()
    rows = []

    for source in sources:
        parts = splitter.split_documents([source])

        for position, chunk in enumerate(parts):
            token_count = len(
                tokenizer.encode(chunk.page_content)
            )

            if token_count > tokenizer.model_max_length:
                raise ValueError(
                    "Chunk exceeds embedding model token limit"
                )

            identity = (
                f"{source.metadata['source_id']}|"
                f"{position}|{chunk.page_content}"
            )
            chunk_id = hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()

            chunk.metadata.update(
                chunk_id=chunk_id,
                chunk_index=position,
                token_count=token_count,
            )

            rows.append(
                {
                    "id": chunk_id,
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata,
                }
            )

    payload = {
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "chunks": rows,
    }

    CHUNKS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    CHUNKS_FILE.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    namespace, chunks = load_snapshot()

    print(
        f"Sources: {len(sources)} | "
        f"Chunks: {len(chunks)}"
    )
    print(f"Snapshot: {CHUNKS_FILE}")
    print(f"Pinecone namespace: {namespace}")


if __name__ == "__main__":
    build_snapshot()