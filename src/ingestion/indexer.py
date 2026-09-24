import time

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config.settings import settings
from src.ingestion.chunker import load_snapshot
from src.retrieval.embeddings import get_embeddings


def connect_index(create_if_missing: bool = False):
    api_key = (
        settings.pinecone_api_key.get_secret_value()
    )

    if not api_key:
        raise ValueError(
            "Set PINECONE_API_KEY in .env"
        )

    client = Pinecone(api_key=api_key)
    name = settings.pinecone_index_name

    if not client.has_index(name):
        if not create_if_missing:
            raise RuntimeError(
                "Index missing; run "
                "src.ingestion.indexer first"
            )

        client.create_index(
            name=name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )

    deadline = time.monotonic() + 120

    while True:
        description = client.describe_index(name)

        if (
            description.dimension
            != settings.embedding_dimension
            or description.metric != "cosine"
        ):
            raise ValueError(
                "Existing index has incompatible "
                "dimensions or metric. Choose a different "
                "PINECONE_INDEX_NAME; do not delete it."
            )

        if description.status["ready"]:
            return client.Index(
                host=description.host
            )

        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Index not ready yet; retry later"
            )

        time.sleep(2)


def main() -> None:
    namespace, chunks = load_snapshot()
    embeddings = get_embeddings()

    actual_dimension = len(
        embeddings.embed_query("dimension check")
    )

    if actual_dimension != settings.embedding_dimension:
        raise ValueError(
            f"Model produces {actual_dimension} dimensions"
        )

    index = connect_index(create_if_missing=True)

    store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=namespace,
    )

    store.add_documents(
        documents=chunks,
        ids=[
            doc.metadata["chunk_id"]
            for doc in chunks
        ],
        batch_size=32,
    )

    print(
        f"Upserted {len(chunks)} chunks into {namespace}"
    )
    print(
        "New records may take a short time "
        "to become queryable."
    )


if __name__ == "__main__":
    main()