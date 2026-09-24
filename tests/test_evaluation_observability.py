import os
os.environ["LANGSMITH_TRACING"] = "false"
import json
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from prometheus_client import generate_latest

from src.api import create_app
from src.evaluation.dataset import load_cases
from src.evaluation.generation_metrics import structural_metrics
from src.evaluation.retrieval_metrics import ranking_metrics
from src.evaluation.workflow_metrics import abstention_metrics, percentile
from src.evaluation.run import summarize, gate_failures
from src.generation.schemas import GenerationResult, GroundedAnswer, abstention
from src.observability.metrics import Telemetry, estimate_cost
from src.observability.runtime import ObservedRuntime
from src.retrieval.corpus import CorpusSnapshot


def test_rank_metrics_known_answer_and_duplicates():
    result = ranking_metrics(["bad", "a", "a", "b"], {"a": 3, "b": 1}, 2)
    assert result["precision"] == .5
    assert result["recall"] == .5
    assert result["mrr"] == .5
    assert 0 < result["ndcg"] < 1
    perfect = ranking_metrics(["a", "a", "b"], {"a": 3, "b": 1}, 2)
    assert perfect == {"precision": 1, "recall": 1, "mrr": 1, "ndcg": 1}


def test_precision_denominator_and_no_gold():
    assert ranking_metrics(["a"], {"a": 3}, 2)["precision"] == .5
    assert ranking_metrics([], {"a": 3}, 2)["recall"] == 0
    assert ranking_metrics(["a"], {}, 2)["recall"] is None


def test_error_is_not_a_correct_abstention():
    result = abstention("Failed")
    result.status = "error"
    assert structural_metrics(result, False, set())["answerability_correct"] == 0
    assert structural_metrics(result, False, set())["citation_id_validity"] is None


def test_citation_id_checks_include_context_membership():
    answer = GroundedAnswer(answerable=True, claims=[{"text":"x", "citations":["C1"]}], missing_information=[])
    result = GenerationResult("answered", answer, sources={"C1":{"chunk_id":"outside"}})
    assert structural_metrics(result, True, {"inside"})["citation_id_validity"] == 0


def test_abstention_precision_recall_errors_count_as_misses():
    rows = [{"expected_answerable":False, "status":"abstained"},
            {"expected_answerable":False, "status":"error"},
            {"expected_answerable":True, "status":"abstained"}]
    assert abstention_metrics(rows) == {"abstention_precision":.5, "abstention_recall":.5}
    assert percentile([1,2,3,4], .95) == 4


def test_summary_excludes_judge_time():
    rows = [{"id":"x", "status":"answered", "elapsed_ms":10000, "pipeline_elapsed_ms":200,
             "expected_answerable":True, "generation":{"answerability_correct":1, "output_valid":1,
             "citation_id_validity":1}, "rankings":{}}]
    summary = summarize(rows)
    assert summary["p95_ms"] == 200
    assert summary["functional_success_rate"] == 1
    assert any("requires --judge" in error for error in gate_failures(summary, "generator", False))


def test_cost_missing_is_not_zero_and_cached_usage_not_mispriced():
    usage = {"input_tokens":1000,"output_tokens":100}
    assert estimate_cost(usage, None, None) is None
    assert estimate_cost(usage, 1, 5) == pytest.approx(.0015)
    assert estimate_cost({**usage,"input_token_details":{"cache_read":100}},1,5) is None


class FakeGraph:
    def invoke(self, inputs, config):
        return {"status":"abstained", "generation":abstention("No evidence"),
                "candidates":[], "warnings":[], "metrics":{"generation_ms":0}}


def test_runtime_records_abstention_and_bounded_labels():
    runtime = ObservedRuntime(graph=FakeGraph())
    state = runtime.invoke({"query":"PRIVATE QUERY VALUE"}, request_id="test-run")
    metrics = generate_latest(runtime.telemetry.registry).decode()
    assert state["request_id"] == "test-run"
    assert 'rag_requests_total{status="abstained"} 1.0' in metrics
    assert "PRIVATE QUERY VALUE" not in metrics
    assert 'rag_requests_in_progress 0.0' in metrics
    assert 'rag_empty_retrieval_total 1.0' in metrics


def test_runtime_exception_is_counted_and_rethrown():
    class Broken:
        def invoke(self, *args, **kwargs):
            raise TimeoutError("simulated")
    runtime = ObservedRuntime(graph=Broken())
    with pytest.raises(TimeoutError):
        runtime.invoke({"query":"x"})
    metrics = generate_latest(runtime.telemetry.registry).decode()
    assert 'rag_requests_total{status="error"} 1.0' in metrics
    assert 'outcome="error",stage="workflow"' in metrics


def test_invalid_output_marks_stage_error():
    telemetry = Telemetry()
    call = telemetry.instrument("workflow", lambda: {"status":"error"})
    call()
    assert 'outcome="error",stage="workflow"' in generate_latest(telemetry.registry).decode()


def test_api_ask_and_metrics_and_request_validation():
    with TestClient(create_app(ObservedRuntime(graph=FakeGraph()))) as client:
        response = client.post("/ask", json={"query":"Why?", "service":"orders-consumer"})
        assert response.status_code == 200
        assert response.json()["status"] == "abstained"
        assert client.get("/metrics").status_code == 200
        assert "rag_requests_total" in client.get("/metrics").text
        assert client.post("/ask", json={"query":"", "environment":"production"}).status_code == 422
        assert client.post("/ask", json={"query":"Why?", "environment":"unknown"}).status_code == 422


def test_api_failure_does_not_expose_exception_text():
    class Broken:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("sensitive original error body")
    with TestClient(create_app(ObservedRuntime(graph=Broken()))) as client:
        response = client.post("/ask", json={"query":"Why?"})
        assert response.status_code == 502
        assert "sensitive original" not in response.text
        assert "request_id" in response.json()["detail"]


def test_dataset_rejects_out_of_scope_gold(tmp_path):
    doc = Document(page_content="x", metadata={"source_id":"a", "approval_status":"draft",
        "environment":"production", "service":"orders-consumer"})
    dataset = [{"id":"x", "query":"why", "scope":{"service":"orders-consumer"},
        "expected_answerable":True, "relevant_sources":{"a":3}, "context_source_ids":["a"],
        "reference_answer":"x"}]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(dataset))
    with pytest.raises(ValueError, match="violates case filters"):
        load_cases(path, CorpusSnapshot("test", [doc]))


def test_keyword_eval_cli_without_cloud(tmp_path, monkeypatch):
    import src.evaluation.run as runner
    metadata = {"source_id":"INC-A", "chunk_id":"a", "title":"Kafka", "approval_status":"approved",
                "environment":"production", "service":"orders-consumer", "technology":"kafka", "document_type":"incident"}
    snapshot = CorpusSnapshot("test", [Document(page_content="Kafka consumer poll expiration", metadata=metadata)])
    monkeypatch.setattr(runner, "load_snapshot", lambda: snapshot)
    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps([{"id":"x", "query":"Kafka", "scope":{"service":"orders-consumer"},
        "expected_answerable":True, "relevant_sources":{"INC-A":3}, "context_source_ids":["INC-A"],
        "reference_answer":"poll expiration"}]))
    output = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["eval", "--stage","keyword","--dataset",str(dataset),"--output",str(output),"--gate"])
    runner.main()
    report = json.loads(output.read_text())
    assert report["complete"]
    assert report["gate"]["passed"]
    assert report["summary"]["ranking"]["keyword"]["recall"] == 1
