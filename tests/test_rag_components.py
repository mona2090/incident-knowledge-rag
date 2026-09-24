"""Offline regression tests. No cloud calls or model downloads."""
import os
os.environ["LANGSMITH_TRACING"] = "false"

from types import SimpleNamespace
import pytest
from langchain_core.documents import Document

from src.generation.generator import Generator, prepare_context
from src.generation.schemas import GroundedAnswer, abstention
from src.reranking.reranker import Reranker
from src.retrieval.corpus import CorpusSnapshot
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.filters import SearchScope
from src.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from src.retrieval.keyword_retriever import KeywordRetriever, tokenize
from src.retrieval.types import HybridResult, SearchHit
from src.workflow.graph import build_graph


def document(identity, text="Kafka lag poll interval", **overrides):
    metadata = {"chunk_id": identity, "source_id": identity, "title": identity,
                "service": "orders-consumer", "environment": "production",
                "technology": "kafka", "document_type": "incident",
                "approval_status": "approved", "source_path": "data/raw/test.json"}
    metadata.update(overrides)
    return Document(id=identity, page_content=text, metadata=metadata)


@pytest.fixture
def snapshot():
    return CorpusSnapshot("test-snapshot", [
        document("approved", "Kafka consumer max.poll.interval.ms expired"),
        document("runbook", "Kafka consumer lag investigation", document_type="runbook"),
        document("draft", "Kafka consumer lag " * 20, approval_status="draft"),
        document("staging", "Kafka consumer lag " * 20, environment="staging"),
        document("payments", "Kafka consumer lag " * 20, service="payments-api"),
    ])


SCOPE = SearchScope(service="orders-consumer")


def test_keyword_filters_before_top_k(snapshot):
    hits = KeywordRetriever(snapshot).search("Kafka consumer", scope=SCOPE, k=2)
    assert {hit.chunk_id for hit in hits} == {"approved", "runbook"}


def test_keyword_preserves_identifiers_and_handles_no_overlap(snapshot):
    assert "max.poll.interval.ms" in tokenize("MAX.POLL.INTERVAL.MS")
    retriever = KeywordRetriever(snapshot)
    assert retriever.search("max.poll.interval.ms", scope=SCOPE)[0].chunk_id == "approved"
    assert retriever.search("zzzznonexistent", scope=SCOPE) == []


def test_empty_scope_is_empty_not_broadened(snapshot):
    assert KeywordRetriever(snapshot).search("Kafka", scope=SearchScope(service="missing")) == []


def test_dense_sends_same_filters(snapshot):
    class Store:
        def similarity_search_with_score(self, **kwargs):
            assert kwargs["filter"] == SCOPE.pinecone_filter()
            return [(snapshot.documents[0], 0.9)]
    assert DenseRetriever(snapshot, store=Store()).search("Kafka", service="orders-consumer")


def test_dense_rejects_scope_violation(snapshot):
    class Store:
        def similarity_search_with_score(self, **kwargs):
            return [(snapshot.documents[2], 0.99)]
    with pytest.raises(RuntimeError):
        DenseRetriever(snapshot, store=Store()).search("Kafka", service="orders-consumer")


def test_rrf_uses_ranks_deduplicates_and_keeps_component_scores():
    a, b, c = [SearchHit(document(identity), {"dense": score})
               for identity, score in [("a", 0.9), ("b", 0.8), ("c", 0.1)]]
    kb = SearchHit(b.document, {"bm25": 50.0})
    result = reciprocal_rank_fusion({"dense": [a, a, b], "keyword": [kb, c]})
    assert result[0].chunk_id == "b"
    assert result[0].scores["rrf"] == pytest.approx(1/62 + 1/61)
    assert result[0].scores["dense"] == 0.8
    assert result[0].scores["bm25"] == 50.0
    assert len(result) == 3


def test_dense_outage_falls_back_without_broadening(snapshot):
    class OfflineDense:
        def search(self, *args, **kwargs):
            raise TimeoutError("simulated")
    result = HybridRetriever(snapshot, dense=OfflineDense()).search("Kafka", scope=SCOPE)
    assert result.warnings
    assert {hit.chunk_id for hit in result.candidates} == {"approved", "runbook"}
    assert result.dense_hits == []


def test_unknown_remote_chunk_rejected(snapshot):
    class WrongSnapshot:
        def search(self, *args, **kwargs):
            return [(document("unknown"), 0.99)]
    with pytest.raises(RuntimeError):
        HybridRetriever(snapshot, dense=WrongSnapshot()).search("Kafka", scope=SCOPE)


def test_reranker_reorders_without_losing_metadata():
    a, b = SearchHit(document("a")), SearchHit(document("b"))
    model = SimpleNamespace(predict=lambda *args, **kwargs: [-2.0, 3.0])
    result = Reranker(model).rerank("Kafka", [a, b], top_n=1)
    assert result[0].chunk_id == "b"
    assert result[0].document.metadata["source_id"] == "b"
    assert a.scores == {}


def test_reranker_rejects_nan():
    model = SimpleNamespace(predict=lambda *args, **kwargs: [float("nan")])
    with pytest.raises(ValueError):
        Reranker(model).rerank("Kafka", [SearchHit(document("a"))])


class FakeStructured:
    def __init__(self, citation="C1", answerable=True):
        self.citation, self.answerable, self.calls = citation, answerable, 0

    def invoke(self, messages):
        self.calls += 1
        answer = GroundedAnswer(answerable=self.answerable,
            claims=[{"text": "The past incident involved poll expiration.", "citations": [self.citation]}]
            if self.answerable else [], missing_information=["Current logs"])
        return {"parsed": answer, "parsing_error": None,
                "raw": SimpleNamespace(usage_metadata={"input_tokens": 20, "output_tokens": 10},
                                       response_metadata={"stop_reason": "tool_use"})}


def test_generator_valid_citations_and_usage():
    result = Generator(FakeStructured()).generate("Kafka?", [SearchHit(document("a"))])
    assert result.status == "answered"
    assert result.sources["C1"]["chunk_id"] == "a"
    assert result.usage["input_tokens"] == 20


def test_generator_rejects_invented_citations():
    result = Generator(FakeStructured("C99")).generate("Kafka?", [SearchHit(document("a"))])
    assert result.status == "error"
    assert result.answer.claims == []
    assert result.warnings


def test_generator_empty_context_skips_api():
    model = FakeStructured()
    assert Generator(model).generate("Kafka?", []).status == "abstained"
    assert model.calls == 0


def test_context_budget_never_truncates_a_chunk():
    small = SearchHit(document("small", "Short useful evidence"))
    large = SearchHit(document("large", "x" * 3000))
    context, sources, ids = prepare_context([large, small], max_chars=500)
    assert ids == ["small"]
    assert sources["C1"]["chunk_id"] == "small"
    assert len(context) <= 500


def test_generator_can_abstain_despite_nonempty_retrieval():
    result = Generator(FakeStructured(answerable=False)).generate("unrelated?", [SearchHit(document("a"))])
    assert result.status == "abstained"
    assert result.sources == {}


def test_graph_empty_route_skips_reranker_and_llm():
    class Empty:
        def search(self, *args, **kwargs):
            return HybridResult([], [], [])
    class MustNotRun:
        def rerank(self, *args, **kwargs):
            pytest.fail("reranker called for empty evidence")
        def generate(self, *args, **kwargs):
            pytest.fail("LLM called for empty evidence")
    result = build_graph(hybrid=Empty(), reranker=MustNotRun(), generator=MustNotRun()).invoke({"query": "Kafka?"})
    assert result["status"] == "abstained"
    assert result["metrics"]["generation_ms"] == 0


def test_graph_reranker_failure_keeps_fused_order_and_warns():
    hits = [SearchHit(document("a")), SearchHit(document("b"))]
    class Retrieval:
        def search(self, *args, **kwargs):
            return HybridResult(hits, hits, [])
    class BrokenReranker:
        def rerank(self, *args, **kwargs):
            raise RuntimeError("simulated native-model setup failure")
    result = build_graph(hybrid=Retrieval(), reranker=BrokenReranker(), generator=Generator(FakeStructured())).invoke({"query": "Kafka?"})
    assert result["status"] == "answered"
    assert result["reranked"][0].chunk_id == "a"
    assert "preserving RRF order" in result["warnings"][0]


def test_generation_errors_are_not_reported_as_abstentions():
    class Retrieval:
        def search(self, *args, **kwargs):
            hits = [SearchHit(document("a"))]
            return HybridResult(hits, hits, [])
    class Ranked:
        def rerank(self, query, hits):
            return hits
    class BrokenGenerator:
        def generate(self, *args):
            raise TimeoutError("simulated provider timeout")
    with pytest.raises(TimeoutError):
        build_graph(hybrid=Retrieval(), reranker=Ranked(), generator=BrokenGenerator()).invoke({"query": "Kafka?"})
