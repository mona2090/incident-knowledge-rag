from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"
load_dotenv(PROJECT_ROOT / ".env", override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)
    pinecone_api_key: SecretStr = SecretStr("")
    pinecone_index_name: str = "incident-knowledge-rag"
    pinecone_namespace: str = "incident-assistant"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = Field(default=384, gt=0)
    chunk_size: int = Field(default=256, gt=0)
    chunk_overlap: int = Field(default=40, ge=0)
    dense_top_k: int = Field(default=10, ge=1, le=100)
    keyword_top_k: int = Field(default=10, ge=1, le=100)
    hybrid_top_k: int = Field(default=10, ge=1, le=200)
    rerank_top_n: int = Field(default=5, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    anthropic_api_key: SecretStr = SecretStr("")
    anthropic_model: str = "claude-haiku-4-5-20251001"
    generation_max_tokens: int = Field(default=1500, ge=256, le=8192)
    generation_timeout_seconds: float = Field(default=60, gt=0)
    context_max_chars: int = Field(default=16000, ge=1000, le=100000)

    @model_validator(mode="after")
    def check_limits(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.rerank_top_n > self.hybrid_top_k:
            raise ValueError("RERANK_TOP_N must not exceed HYBRID_TOP_K")
        return self


settings = Settings()
