import json
import logging
import math
import os
from functools import wraps
from time import perf_counter

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from src.observability.tracing import traced

LOGGER = logging.getLogger("incident_rag.operations")
if not LOGGER.handlers:
    LOGGER.addHandler(logging.StreamHandler())
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def estimate_cost(usage, input_rate, output_rate):
    if input_rate is None or output_rate is None or not usage:
        return None
    details = usage.get("input_token_details", {}) or {}
    if details.get("cache_read", 0) or details.get("cache_creation", 0):
        return None  # Different cache prices require separate accounting.
    if "input_tokens" not in usage or "output_tokens" not in usage:
        return None
    return (usage["input_tokens"] * input_rate + usage["output_tokens"] * output_rate) / 1_000_000


class Telemetry:
    def __init__(self, registry=None):
        self.registry = registry if registry is not None else CollectorRegistry()
        self.requests = Counter("rag_requests_total", "Completed accepted RAG requests", ["status"], registry=self.registry)
        self.duration = Histogram("rag_request_duration_seconds", "RAG wall time including local queue wait",
            buckets=(.1,.5,1,2,5,10,20,30,60,120,300), registry=self.registry)
        self.stages = Histogram("rag_stage_duration_seconds", "Stage elapsed seconds", ["stage", "outcome"],
            buckets=(.01,.05,.1,.5,1,2,5,10,30,60,120,300), registry=self.registry)
        self.fallbacks = Counter("rag_fallbacks_total", "Fallback activations", ["component"], registry=self.registry)
        self.empty = Counter("rag_empty_retrieval_total", "Requests with zero candidates", registry=self.registry)
        self.tokens = Counter("rag_tokens_total", "Observed generator tokens, excluding judge calls", ["direction"], registry=self.registry)
        self.cost = Counter("rag_estimated_generation_cost_usd_total", "Optional uncached generation cost estimate", registry=self.registry)
        self.cost_missing = Counter("rag_generation_cost_unavailable_total", "Generation results lacking a cost estimate", registry=self.registry)
        self.inflight = Gauge("rag_requests_in_progress", "Accepted requests running or waiting locally", registry=self.registry)
        self.candidates = Histogram("rag_candidate_count", "Fused candidate count", buckets=(0,1,2,5,10,20,50,100), registry=self.registry)
        def rate(name):
            value = os.getenv(name, "").strip()
            parsed = float(value) if value else None
            if parsed is not None and (not math.isfinite(parsed) or parsed < 0):
                raise ValueError(f"{name} must be a nonnegative finite number")
            return parsed
        self.input_rate = rate("RAG_INPUT_USD_PER_MILLION")
        self.output_rate = rate("RAG_OUTPUT_USD_PER_MILLION")

    def instrument(self, stage, function):
        @wraps(function)
        def measured(*args, **kwargs):
            start = perf_counter()
            outcome = "error"
            try:
                result = function(*args, **kwargs)
                status = result.get("status") if isinstance(result, dict) else getattr(result, "status", None)
                outcome = "error" if status == "error" else "ok"
                return result
            finally:
                self.stages.labels(stage, outcome).observe(perf_counter() - start)
        return traced(measured, f"rag.{stage}")

    def complete(self, state):
        generation = state["generation"]
        usage = generation.usage
        for direction in ("input", "output"):
            count = usage.get(f"{direction}_tokens")
            if isinstance(count, (int, float)) and count >= 0:
                self.tokens.labels(direction).inc(count)
        cost = estimate_cost(usage, self.input_rate, self.output_rate)
        if cost is not None:
            self.cost.inc(cost)
        elif generation.context_chunk_ids:
            self.cost_missing.inc()
        self.candidates.observe(len(state["candidates"]))
        if not state["candidates"]:
            self.empty.inc()
        for warning in state["warnings"]:
            if warning.startswith("Dense retrieval unavailable"):
                self.fallbacks.labels("dense").inc()
            elif warning.startswith("Reranker unavailable"):
                self.fallbacks.labels("reranker").inc()
        return cost
