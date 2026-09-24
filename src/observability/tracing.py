import os
from src.config.settings import settings  # Loads .env before SDK configuration.

# These defaults hide captured content, including nested LangChain/LangGraph runs.
os.environ.setdefault("LANGSMITH_HIDE_INPUTS", "true")
os.environ.setdefault("LANGSMITH_HIDE_OUTPUTS", "true")

from langsmith import traceable, tracing_context


def tracing_enabled():
    return os.getenv("LANGSMITH_TRACING", "false").lower() in {"true", "1"}


def traced(function, name):
    return traceable(name=name, run_type="chain", 
    )(function)


def request_context(request_id):
    return tracing_context(enabled=tracing_enabled(),
        project_name=os.getenv("LANGSMITH_PROJECT", "incident-knowledge-rag"),
        metadata={"request_id": request_id, "pipeline_version": (os.getenv("PIPELINE_VERSION") or "incident-rag-v1")})
