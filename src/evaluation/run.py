import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from src.config.settings import PROJECT_ROOT, settings
from src.observability import tracing  # Apply trace privacy defaults for every eval stage.
from src.generation.prompts import SYSTEM_PROMPT
from src.evaluation.dataset import load_cases
from src.evaluation.generation_metrics import structural_metrics
from src.evaluation.retrieval_metrics import filter_violations, ranking_metrics, unique_source_ids
from src.evaluation.workflow_metrics import abstention_metrics, average, percentile
from src.retrieval.corpus import load_snapshot
from src.retrieval.types import SearchHit


def summarize(rows):
    summary = {"cases": len(rows), "errors": sum(row.get("status") == "error" for row in rows),
               "degraded_cases": sum(bool(row.get("warnings")) for row in rows),
               "judge_errors": sum("judge_error" in row for row in rows),
               "filter_violations": sum(row.get("filter_violations", 0) for row in rows),
               "p50_ms": percentile([row.get("pipeline_elapsed_ms", row["elapsed_ms"]) for row in rows], .50),
               "p95_ms": percentile([row.get("pipeline_elapsed_ms", row["elapsed_ms"]) for row in rows], .95)}
    branches = sorted({name for row in rows for name in row.get("rankings", {})})
    summary["ranking"] = {branch: {
        metric: average([row.get("rankings", {}).get(branch, {}).get(metric) for row in rows])
        for metric in ("precision", "recall", "mrr", "ndcg")
    } for branch in branches}
    deltas = [row["rankings"]["reranked"]["ndcg"] - row["rankings"]["hybrid"]["ndcg"]
              for row in rows if row.get("rankings", {}).get("reranked", {}).get("ndcg") is not None]
    summary["reranker_ndcg_delta"] = average(deltas)
    summary["generation"] = {metric: average([row.get("generation", {}).get(metric) for row in rows])
        for metric in ("answerability_correct", "output_valid", "claim_citation_coverage", "citation_id_validity")}
    if any("generation" in row or "answer" in row for row in rows):
        summary.update(abstention_metrics(rows))
        # Errors count as failure, including cases absent from the generation-metric mean.
        summary["functional_success_rate"] = sum(
            row.get("generation", {}).get("answerability_correct", 0) == 1 and
            row.get("generation", {}).get("output_valid", 0) == 1 and
            row.get("generation", {}).get("citation_id_validity") in (None, 1.0)
            for row in rows) / len(rows)
    summary["judge"] = {metric: average([row.get("judge", {}).get(metric) for row in rows])
        for metric in ("correctness", "faithfulness", "citation_support")}
    summary["judged_cases"] = sum("judge" in row for row in rows)
    summary["generator_tokens"] = {name: sum(row.get("usage", {}).get(name, 0) for row in rows)
                                   for name in ("input_tokens", "output_tokens")}
    return summary


def gate_failures(summary, stage, use_judge, min_recall=.8, min_quality=.8):
    failed = []
    for metric in ("errors", "degraded_cases", "judge_errors", "filter_violations"):
        if summary[metric]:
            failed.append(f"{metric} must be zero")
    if stage in {"keyword", "retrieval", "reranker", "workflow"}:
        branch = {"keyword": "keyword", "retrieval": "hybrid", "reranker": "reranked", "workflow": "reranked"}[stage]
        value = summary["ranking"].get(branch, {}).get("recall")
        if value is None or value < min_recall:
            failed.append(f"{branch} recall below threshold or not evaluable")
    if stage in {"generator", "workflow"}:
        if summary.get("functional_success_rate", 0) < 1:
            failed.append("functional_success_rate must be 1 on the starter dataset")
        if not use_judge or summary["judged_cases"] == 0:
            failed.append("Semantic gate requires --judge and at least one judged answer")
        for metric in ("correctness", "faithfulness", "citation_support"):
            value = summary["judge"][metric]
            if use_judge and (value is None or value < min_quality):
                failed.append(f"judge {metric} below threshold or not evaluable")
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["keyword", "retrieval", "reranker", "generator", "workflow"])
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/evals/incident_eval_v1.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/eval_latest.json")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--min-recall", type=float, default=.8)
    parser.add_argument("--min-quality", type=float, default=.8)
    args = parser.parse_args()
    if args.k < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--k and --limit must be positive")
    if not 0 <= args.min_recall <= 1 or not 0 <= args.min_quality <= 1:
        parser.error("Thresholds must lie between 0 and 1")
    if args.judge and args.stage not in {"generator", "workflow"}:
        parser.error("--judge is supported for generator and workflow evaluations")
    snapshot = load_snapshot()
    cases = load_cases(args.dataset, snapshot)
    if args.limit:
        cases = cases[:args.limit]
    by_id = {doc.metadata["chunk_id"]: doc for doc in snapshot.documents}
    keyword = hybrid = reranker = generator = runtime = judge = None
    if args.stage == "keyword":
        from src.retrieval.keyword_retriever import KeywordRetriever
        keyword = KeywordRetriever(snapshot)
    if args.stage in {"retrieval", "reranker"}:
        from src.retrieval.hybrid_retriever import HybridRetriever
        hybrid = HybridRetriever(snapshot)
    if args.stage == "reranker":
        from src.reranking.reranker import Reranker
        reranker = Reranker()
    if args.stage == "generator":
        from src.generation.generator import Generator
        generator = Generator()
    if args.stage == "workflow":
        from src.observability.runtime import ObservedRuntime
        runtime = ObservedRuntime(snapshot=snapshot)
    if args.judge:
        from src.evaluation.judge import Judge
        judge = Judge()

    rows = []
    manifest = {"dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                "snapshot_namespace": snapshot.namespace, "stage": args.stage, "k": args.k,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "generator_model": settings.anthropic_model, "reranker_model": settings.reranker_model,
                "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                "context_max_chars": settings.context_max_chars,
                "generation_max_tokens": settings.generation_max_tokens,
                "embedding_model": settings.embedding_model,
                "top_k": {"dense": settings.dense_top_k, "keyword": settings.keyword_top_k,
                          "hybrid": settings.hybrid_top_k, "rerank": settings.rerank_top_n},
                "rrf_k": settings.rrf_k, "judge_model": judge.model_name if judge else None,
                "judge_version": judge.VERSION if judge else None,
                "packages": {name: version(name) for name in ("langgraph", "langchain-anthropic", "rank-bm25")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases):
        start = perf_counter()
        row = {"id": case.id, "tags": case.tags, "expected_answerable": case.expected_answerable,
               "first_case_in_process": index == 0, "status": "ok", "warnings": [], "rankings": {}}
        branches, generated = {}, None
        try:
            if keyword:
                branches["keyword"] = keyword.search(case.query, scope=case.scope, k=max(args.k, settings.keyword_top_k))
            elif hybrid:
                retrieval = hybrid.search(case.query, scope=case.scope)
                branches = {"dense": retrieval.dense_hits, "keyword": retrieval.keyword_hits, "hybrid": retrieval.candidates}
                row["warnings"] = retrieval.warnings
                if reranker:
                    branches["reranked"] = reranker.rerank(case.query, retrieval.candidates)
            elif generator:
                # Fixed, labeled evidence isolates generation from retrieval mistakes.
                context = [SearchHit(doc) for source in case.context_source_ids for doc in snapshot.documents
                           if doc.metadata["source_id"] == source and case.scope.allows(doc.metadata)]
                generated = generator.generate(case.query, context)
            elif runtime:
                state = runtime.invoke({"query": case.query, **case.scope.model_dump()})
                branches = {"dense": state["dense_hits"], "keyword": state["keyword_hits"],
                            "hybrid": state["candidates"], "reranked": state["reranked"]}
                generated = state["generation"]
                row.update(warnings=state["warnings"], request_id=state["request_id"], metrics=state["metrics"])
            row["filter_violations"] = sum(filter_violations(hits, case.scope) for hits in branches.values())
            for branch, hits in branches.items():
                ids = unique_source_ids(hits)
                row["rankings"][branch] = {**ranking_metrics(ids, case.relevant_sources, args.k), "source_ids": ids}
            if generated:
                actual_context = [by_id[chunk_id] for chunk_id in generated.context_chunk_ids]
                allowed = {doc.metadata["chunk_id"] for doc in actual_context if case.scope.allows(doc.metadata)}
                row["filter_violations"] += sum(not case.scope.allows(doc.metadata) for doc in actual_context)
                row.update(status=generated.status, answer=generated.answer.model_dump(), sources=generated.sources,
                           context_chunk_ids=generated.context_chunk_ids, usage=generated.usage,
                           generation=structural_metrics(generated, case.expected_answerable, allowed))
                row["warnings"] = list(dict.fromkeys(row["warnings"] + generated.warnings))
                row["pipeline_elapsed_ms"] = round((perf_counter() - start) * 1000, 2)
                if judge and generated.status == "answered":
                    judge_started = perf_counter()
                    try:
                        row["judge"] = judge.score(case, generated, actual_context)
                    except Exception as exc:
                        row["judge_error"] = type(exc).__name__
                    row["judge_ms"] = round((perf_counter() - judge_started) * 1000, 2)
        except Exception as exc:
            row.update(status="error", error_type=type(exc).__name__)
        row["elapsed_ms"] = round((perf_counter() - start) * 1000, 2)
        rows.append(row)
        print(f"{case.id}: {row['status']} ({row['elapsed_ms']} ms)", flush=True)
        # Keep completed cases if a later call fails or the process is interrupted.
        args.output.write_text(json.dumps({"manifest": manifest, "rows": rows, "complete": False}, indent=2))
    summary = summarize(rows)
    failures = gate_failures(summary, args.stage, args.judge, args.min_recall, args.min_quality) if args.gate else []
    report = {"manifest": manifest, "rows": rows, "summary": summary, "complete": True,
              "gate": {"requested": args.gate, "passed": not failures if args.gate else None, "failures": failures}}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))
    print("Report:", args.output)
    if failures:
        print("Gate failures:", failures)
    if summary["errors"] or summary["judge_errors"] or failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
