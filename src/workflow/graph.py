from time import perf_counter

from src.config.settings import settings
from langgraph.graph import END, START, StateGraph
from src.generation.generator import Generator
from src.generation.schemas import abstention
from src.reranking.reranker import Reranker
from src.retrieval.filters import SearchScope, clean_query
from src.retrieval.hybrid_retriever import HybridRetriever
from src.workflow.state import RAGState


def build_graph(*, hybrid=None, reranker=None, generator=None):
    hybrid = hybrid if hybrid is not None else HybridRetriever()
    reranker = reranker if reranker is not None else Reranker()
    generator = generator if generator is not None else Generator()

    def validate(state: RAGState):
        scope = SearchScope(
            service=state.get("service"), environment=state.get("environment", "production"),
            technology=state.get("technology"), document_type=state.get("document_type"),
        )
        # Reset run-local data. This graph has no chat memory/checkpointer.
        return {"query": clean_query(state["query"]), "scope": scope,
                "warnings": [], "metrics": {}, "dense_hits": [], "keyword_hits": [],
                "candidates": [], "reranked": [], "status": "running"}

    def retrieve(state: RAGState):
        started = perf_counter()
        result = hybrid.search(state["query"], scope=state["scope"])
        # Enforce the same scope at the final retrieval boundary as well.
        for hit in result.candidates:
            if not state["scope"].allows(hit.document.metadata):
                raise RuntimeError("Candidate violated metadata scope")
        return {"candidates": result.candidates, "dense_hits": result.dense_hits,
                "keyword_hits": result.keyword_hits, "warnings": result.warnings,
                "metrics": {**result.metrics, "retrieval_ms": round((perf_counter() - started) * 1000, 2)}}

    def route_after_retrieval(state: RAGState):
        return "rerank" if state["candidates"] else "abstain"

    def rerank(state: RAGState):
        started = perf_counter()
        warnings = list(state["warnings"])
        try:
            ranked = reranker.rerank(state["query"], state["candidates"])
        except Exception as exc:
            warnings.append(f"Reranker unavailable ({type(exc).__name__}); preserving RRF order.")
            ranked = state["candidates"][:settings.rerank_top_n]
        return {"reranked": ranked, "warnings": warnings,
                "metrics": {**state["metrics"], "rerank_ms": round((perf_counter() - started) * 1000, 2),
                            "reranked_count": len(ranked)}}

    def generate(state: RAGState):
        started = perf_counter()
        # Provider/authentication errors propagate; they are NOT no-evidence abstentions.
        result = generator.generate(state["query"], state["reranked"])
        return {"generation": result, "status": result.status,
                "warnings": state["warnings"] + result.warnings,
                "metrics": {**state["metrics"], "generation_ms": round((perf_counter() - started) * 1000, 2),
                            "context_chunk_count": len(result.context_chunk_ids)}}

    def no_evidence(state: RAGState):
        result = abstention("No matching evidence was found. Check the service/environment or add relevant documents.")
        return {"generation": result, "status": result.status,
                "metrics": {**state["metrics"], "generation_ms": 0.0, "context_chunk_count": 0}}

    builder = StateGraph(RAGState)
    builder.add_node("validate", validate)
    builder.add_node("retrieve", retrieve)
    builder.add_node("rerank", rerank)
    builder.add_node("generate", generate)
    builder.add_node("abstain", no_evidence)
    builder.add_edge(START, "validate")
    builder.add_edge("validate", "retrieve")
    builder.add_conditional_edges("retrieve", route_after_retrieval,
                                  {"rerank": "rerank", "abstain": "abstain"})
    builder.add_edge("rerank", "generate")
    builder.add_edge("generate", END)
    builder.add_edge("abstain", END)
    return builder.compile()
