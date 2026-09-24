from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config.settings import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
        },
        query_encode_kwargs={
            "normalize_embeddings": True,
        },
    )