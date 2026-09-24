import json
import threading
from time import perf_counter
from uuid import uuid4

from src.observability.tracing import request_context
from src.observability.metrics import LOGGER, Telemetry
from src.retrieval.corpus import load_snapshot
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.keyword_retriever import KeywordRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.reranker import Reranker
from src.generation.generator import Generator
from src.workflow.graph import build_graph


class ObservedRuntime:
    """Reuse one instance. A lock keeps this local CPU tutorial sequential."""
    def __init__(self, snapshot=None, telemetry=None, graph=None):
        self.telemetry = telemetry if telemetry is not None else Telemetry()
        self.lock = threading.Lock()
        if graph is None:
            snapshot = snapshot if snapshot is not None else load_snapshot()
            def proxy(target, method, stage):
                # A separate proxy preserves the original object's methods.
                class Proxy:
                    pass
                instance = Proxy()
                setattr(instance, method, self.telemetry.instrument(stage, getattr(target, method)))
                return instance
            dense = proxy(DenseRetriever(snapshot), "search", "dense")
            keyword = proxy(KeywordRetriever(snapshot), "search", "keyword")
            hybrid = proxy(HybridRetriever(snapshot, dense=dense, keyword=keyword), "search", "hybrid")
            reranker = proxy(Reranker(), "rerank", "reranker")
            generator = proxy(Generator(), "generate", "generator")
            graph = build_graph(hybrid=hybrid, reranker=reranker, generator=generator)
        self._invoke = self.telemetry.instrument("workflow", graph.invoke)

    def invoke(self, inputs, request_id=None):
        request_id = request_id or str(uuid4())
        start, status = perf_counter(), "error"
        error_type, state = None, None
        self.telemetry.inflight.inc()
        try:
            with self.lock, request_context(request_id):
                state = self._invoke(inputs, config={"recursion_limit": 12})
            status = state["status"]
            cost = self.telemetry.complete(state)
            state["request_id"] = request_id
            state["estimated_generation_usd"] = cost
            state["metrics"]["total_ms"] = round((perf_counter() - start) * 1000, 2)
            return state
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            seconds = perf_counter() - start
            self.telemetry.requests.labels(status).inc()
            self.telemetry.duration.observe(seconds)
            self.telemetry.inflight.dec()
            LOGGER.info(json.dumps({"event": "rag_request", "request_id": request_id,
                "status": status, "duration_ms": round(seconds * 1000, 2),
                "error_type": error_type, "degraded": bool(state and state.get("warnings"))}))


def public_result(state):
    result = state["generation"]
    return {"request_id": state["request_id"], "status": state["status"],
            "answer": result.answer.model_dump(), "sources": result.sources,
            "usage": result.usage, "metrics": state["metrics"], "warnings": state["warnings"],
            "degraded": bool(state["warnings"]),
            "estimated_generation_usd": state.get("estimated_generation_usd")}
