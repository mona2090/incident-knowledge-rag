"""Run each tutorial stage: python -m src.demo --stage keyword --query '...'"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from src.config.settings import settings
from src.retrieval.filters import SearchScope, clean_query
from src.retrieval.types import SearchHit


def hit_dict(hit: SearchHit) -> dict:
    return {"chunk_id": hit.chunk_id, "metadata": hit.document.metadata,
            "text": hit.document.page_content, "scores": hit.scores, "ranks": hit.ranks}


def generation_dict(result) -> dict:
    return {"status": result.status, "answer": result.answer.model_dump(),
            "sources": result.sources, "context_chunk_ids": result.context_chunk_ids,
            "usage": result.usage, "warnings": result.warnings}


def run(args) -> dict:
    query = clean_query(args.query)
    scope = SearchScope(service=args.service, environment=args.environment,
                        technology=args.technology, document_type=args.document_type)
    started = perf_counter()
    if args.stage == "keyword":
        from src.retrieval.keyword_retriever import KeywordRetriever
        hits = KeywordRetriever().search(query, scope=scope)
        return {"stage": args.stage, "hits": [hit_dict(hit) for hit in hits],
                "elapsed_ms": round((perf_counter() - started) * 1000, 2)}

    if args.stage == "workflow":
        from src.workflow.graph import build_graph
        state = build_graph().invoke({"query": query, **scope.model_dump()}, config={"recursion_limit": 12})
        return {**generation_dict(state["generation"]), "stage": args.stage,
                "warnings": state["warnings"], "degraded": bool(state["warnings"]),
                "metrics": {**state["metrics"], "total_ms": round((perf_counter() - started) * 1000, 2)},
                "dense_hits": [hit_dict(hit) for hit in state["dense_hits"]],
                "keyword_hits": [hit_dict(hit) for hit in state["keyword_hits"]],
                "candidates": [hit_dict(hit) for hit in state["candidates"]],
                "reranked": [hit_dict(hit) for hit in state["reranked"]]}

    from src.retrieval.hybrid_retriever import HybridRetriever
    retrieval = HybridRetriever().search(query, scope=scope)
    report = {"stage": args.stage, "warnings": retrieval.warnings, "metrics": retrieval.metrics,
              "dense_hits": [hit_dict(hit) for hit in retrieval.dense_hits],
              "keyword_hits": [hit_dict(hit) for hit in retrieval.keyword_hits],
              "candidates": [hit_dict(hit) for hit in retrieval.candidates]}
    if args.stage == "hybrid":
        return report

    from src.reranking.reranker import Reranker
    rank_started = perf_counter()
    ranked = Reranker().rerank(query, retrieval.candidates)
    report["metrics"]["rerank_ms"] = round((perf_counter() - rank_started) * 1000, 2)
    report["reranked"] = [hit_dict(hit) for hit in ranked]
    if args.stage == "generate":
        from src.generation.generator import Generator
        generation_started = perf_counter()
        generated = Generator().generate(query, ranked)
        report.update(generation_dict(generated))
        report["warnings"] = retrieval.warnings + generated.warnings
        report["metrics"]["generation_ms"] = round((perf_counter() - generation_started) * 1000, 2)
    report["metrics"]["total_ms"] = round((perf_counter() - started) * 1000, 2)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["keyword", "hybrid", "rerank", "generate", "workflow"], required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--service")
    parser.add_argument("--environment", default="production")
    parser.add_argument("--technology")
    parser.add_argument("--document-type", choices=["incident", "runbook"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if "answer" not in report:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    print("Status:", report["status"])
    for claim in report["answer"]["claims"]:
        labels = " ".join(f"[{citation}]" for citation in claim["citations"])
        print(f"- {claim['text']} {labels}")
    for gap in report["answer"]["missing_information"]:
        print("Need:", gap)
    for citation, source in report["sources"].items():
        print(f"[{citation}] {source['source_id']}: {source['title']}")
    for warning in report.get("warnings", []):
        print("WARNING:", warning)
    print("Metrics:", json.dumps(report.get("metrics", {})))
    print("Token usage:", json.dumps(report.get("usage", {})))
    if report["status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
