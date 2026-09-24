# Steps 11–12: Evaluation and observability

Update for the working `incident-knowledge-rag` project through Step 10. This
adds evals and an instrumented runtime; existing retrieval/generation/graph
implementation files remain the same. Use the NEW commands below for tracing
and metrics. The old `src.demo` CLI is still available but does not use the new
Prometheus wrappers.

## Install

Unzip the download. From your existing project root (adjust the download path):

```bash
./.venv/bin/python "$HOME/Downloads/rag-steps-11-12/apply_update.py" --project . --apply
./.venv/bin/python -m pip install -r requirements.steps11-12.txt
./.venv/bin/python -m pip check
```

The installer backs up replaced files under `.rag-step11-12-backups/` and copies
this guide to `docs/steps11-12.md`. It does not edit `.env` or reindex your corpus.
Merge `env.steps11-12.example` into `.env`, avoiding duplicate keys. Keep your
existing working Pinecone, Anthropic, embedding, and reranker settings.
Add `.rag-step11-12-backups/` and generated reports to `.gitignore`.

Retain your Mac troubleshooting settings in the same terminal:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

## Step 11A: Understand the starter dataset

`data/evals/incident_eval_v1.json` contains 10 labeled examples matching the five
synthetic documents from Step 3. Six are answerable; four test missing scope,
missing measurements, or missing current-incident evidence. One includes a user
instruction that tries to bypass approval filtering. This is a starter/dev set,
not a held-out benchmark or an enterprise safety certification.

Each record includes: question, scope, expected answerability, graded relevant
SOURCE IDs, fixed context source IDs, reference answer, and tags. Grades 1/2/3
mean useful/relevant/highly relevant. Review and adapt these judgments for your
use case. The loader rejects missing or out-of-scope gold sources. If you changed
the sample documents, update the labels and references before interpreting scores.

Metrics use document/source IDs, not chunk IDs, because those are the labels in
this starter set. Multiple retrieved chunks from a source count once, retaining
its first rank. Add chunk-level annotations before evaluating chunk relevance.

## Step 11B: Retriever evals

Run a local keyword baseline first (no cloud/model calls):

```bash
./.venv/bin/python -m src.evaluation.run --stage keyword --k 2 --output reports/eval_keyword.json
```

Then compare dense, keyword, and fused rankings (Pinecone calls):

```bash
./.venv/bin/python -m src.evaluation.run --stage retrieval --k 2 --output reports/eval_retrieval.json
```

- Precision@k = relevant retrieved / k. Fewer than k results still use k in the denominator.
- Recall@k = relevant retrieved / total labeled relevant sources.
- MRR@k = reciprocal rank of the first relevant result, or zero.
- nDCG@k = graded discounted gain / ideal graded discounted gain, with gains 2^grade - 1.
- Filter violations count out-of-scope occurrences at evaluated boundaries.

No-gold/unanswerable cases get null ranking metrics, not artificial perfect or
zero relevance scores. They are evaluated for abstention separately. Default k=2
fits this tiny corpus; use larger corpora and appropriate k values for real evals.

## Step 11C: Reranker evals

```bash
./.venv/bin/python -m src.evaluation.run --stage reranker --k 2 --output reports/eval_reranker.json
```

Compare `ranking.hybrid` with `ranking.reranked`, especially
`reranker_ndcg_delta`. It is the mean paired difference on evaluable queries.
Positive means improved order, zero means no measured change, negative means
worse. Few candidates may leave little room for improvement. The reranker can
reorder retrieved evidence but cannot recover evidence absent from its candidates.

## Step 11D: Isolated generator evals

```bash
./.venv/bin/python -m src.evaluation.run --stage generator --output reports/eval_generator.json
```

This provides the labeled fixed context to the generator, bypassing retrieval.
It tests answerability decisions, schema/output validity, claim citation
coverage, and citation-ID membership in the actual context. Abstentions have
null citation metrics. Errors are not credited as correct abstentions.

For semantic evaluation, opt into a Claude judge (additional paid API calls):

```bash
./.venv/bin/python -m src.evaluation.run --stage generator --judge --output reports/eval_generator_judged.json
```

Judge dimensions: correctness against the question/reference, faithfulness to
actual evidence, and support from each claim's specifically cited chunks. The
judge sees only chunks actually packed into the generator context. It evaluates
answered outputs; abstentions are scored against labeled answerability.
Inspect `judged_cases` alongside averages to understand coverage.

Set `EVAL_JUDGE_MODEL` to a model available to your Anthropic API account, or leave
it blank to use `ANTHROPIC_MODEL`. Using the same model for generation and judging
creates correlated errors. Scores are fallible estimates; review samples manually
and calibrate the rubric against human judgments. The judge does not execute tools.

## Step 11E: End-to-end workflow evals

```bash
./.venv/bin/python -m src.evaluation.run --stage workflow --judge --output reports/eval_workflow.json
```

Reports include every retrieval branch, post-rerank rankings, final answers,
source citations, context IDs, failures/degraded modes, stage timings, generator
token usage, and optional judge results. Pipeline latency excludes judge time;
each row also records total case time and judge time. The first case can include
model startup/download costs. Ten cases are insufficient for meaningful p95 SLO
estimation; run a larger repeated workload in one process for operational testing.

Reports contain generated text and document identifiers. Do not commit real
incident reports without applying your data policy. Generator token totals exclude
judge usage and may omit billable work from failed/retried calls. They are not a
complete billing ledger. Partial reports are saved after each completed case.

Manifest: dataset hash, snapshot namespace, model IDs, prompt hash, context budget,
retrieval settings, judge rubric version, package versions, and timestamp. Keep
these constant when comparing changes other than the parameter under test.

### Gates and checks

```bash
./.venv/bin/python -m pytest tests/test_steps06_10.py tests/test_steps11_12.py -q
./.venv/bin/python -m src.evaluation.run --stage workflow --judge --gate --output reports/eval_gate.json
```

The illustrative gate requires zero errors, degraded cases, judge errors and
filter violations; mean final recall >=0.8; correct functional behavior on all
starter cases; and mean judge scores >=0.8. `--min-recall` and `--min-quality`
adjust those thresholds. Generation/workflow gates require a judge and at least
one judged answer; skipping semantics cannot produce a passing semantic gate.
Use `--limit 2` for a cheaper smoke run, not to claim full-dataset coverage.
Calibrate real release thresholds on a separate held-out set. No claim is made
that the live model necessarily passes these starter gates.

## Step 12A: LangSmith traces

Tracing is off by default. To enable it, update `.env`:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-key
LANGSMITH_PROJECT=incident-knowledge-rag
LANGSMITH_HIDE_INPUTS=true
LANGSMITH_HIDE_OUTPUTS=true
```

Then run the instrumented CLI:

```bash
./.venv/bin/python -m src.observability.run --query "What should we check for consumer lag?" --service orders-consumer --environment production
```

In your LangSmith project, inspect `rag.workflow`, LangGraph nodes, `rag.hybrid`,
`rag.dense`, `rag.keyword`, `rag.reranker`, `rag.generator`, and the nested Claude
call. Custom span inputs/outputs are suppressed; hide-input/output settings apply
to nested SDK traces. Request IDs link local JSON logs and trace metadata.
For synthetic-only debugging, nested SDK inputs/outputs can be shown by setting
the hide flags to false and restarting. Errors, metadata, and SDK serialization
can still carry information; hiding input/output fields is not comprehensive
redaction. Review traces before using real enterprise incident data.

JSON operational logs intentionally omit question, source text, answer text, and
exception messages. They include request ID, status, latency, degraded flag, and
exception class. Native segmentation faults bypass Python instrumentation and
need process/host monitoring.

## Step 12B: Serve the API and metrics

```bash
./.venv/bin/python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Keep this terminal running. Use a second terminal:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"What caused the past consumer-lag incident?","service":"orders-consumer","environment":"production"}'

curl http://127.0.0.1:8000/metrics
```

Swagger UI: http://127.0.0.1:8000/docs

This is a LOCAL example: one persistent graph, one worker, and serialized CPU
inference. The lock avoids concurrent initialization and measures local queue
wait in request duration. It is not a throughput-optimized production server.
Health is process liveness, not an upstream Pinecone/Claude health guarantee.
Add authenticated tenant-scoped authorization before enterprise deployment.
The request counter measures accepted RAG requests; HTTP validation failures
(422s), scraping, and health requests are outside that counter.

Metrics:

| Name | Meaning |
|---|---|
| rag_requests_total{status} | Answered, abstained, and failed accepted requests |
| rag_request_duration_seconds | End-to-end histogram, including local queue wait |
| rag_stage_duration_seconds{stage,outcome} | Dense, keyword, hybrid, reranker, generator, workflow |
| rag_fallbacks_total{component} | Dense/BM25-only or reranker/RRF-order fallbacks |
| rag_empty_retrieval_total | Zero-candidate requests |
| rag_candidate_count | Candidate-count histogram |
| rag_tokens_total{direction} | Observed generator input/output tokens |
| rag_requests_in_progress | Running/locally queued requests |
| rag_estimated_generation_cost_usd_total | Optional uncached token-cost estimate |
| rag_generation_cost_unavailable_total | Generations without a usable cost estimate |

Metric labels have bounded stage/outcome values; questions, service names,
document IDs, and request IDs are not Prometheus labels. Histograms/counters
are cumulative within the running process. Do not run the old short-lived demo
CLI and expect its metrics to appear in this API process.

## Step 12C: Prometheus and Grafana

If you already have the Prometheus binary installed, start it from the project root:

```bash
prometheus --config.file=observability/prometheus.yml
```

Prometheus scrapes the local API every 15 seconds. Open http://127.0.0.1:9090.
For a containerized collector, adjust the target to reach the host API; the
provided loopback target assumes Prometheus runs directly on the Mac.

Import `observability/grafana_dashboard.json` into Grafana and select your
Prometheus data source. It includes throughput, outcomes, p95, stage latency,
fallbacks, tokens, and unpriced generation counts.

Useful PromQL:

```promql
sum(rate(rag_requests_total[5m]))
histogram_quantile(0.95, sum by (le) (rate(rag_request_duration_seconds_bucket[5m])))
sum(rate(rag_requests_total{status="error"}[5m])) / clamp_min(sum(rate(rag_requests_total[5m])), 0.001)
sum by (component) (rate(rag_fallbacks_total[5m]))
```

`alerts.yml` provides illustrative rules: sustained error rate >5%, p95 >15s,
and repeated fallbacks. Error/latency alerts require enough recent traffic.
They create alerts in Prometheus; notification delivery requires your own
Alertmanager routes. These thresholds are examples, not established SLOs.

Optional cost estimation: set CURRENT `RAG_INPUT_USD_PER_MILLION` and
`RAG_OUTPUT_USD_PER_MILLION` for your generator model. Empty means unavailable,
not zero cost. Cached-token calls are excluded because cache pricing differs.
This excludes judge calls, Pinecone, infrastructure, and unobserved retry costs.

## Improve metrics using evidence

1. Compare dense vs keyword vs fused Recall/MRR/nDCG on a larger labeled set.
2. Tune candidate counts before reranking; poor recall cannot be fixed by the generator.
3. Compare reranker nDCG gain with its latency and resource cost.
4. Tune chunk boundaries/context selection when citation support or faithfulness is weak.
5. Measure abstention precision/recall, especially on missing-current-evidence cases.
6. Compare warm-process latency and tokens at similar answer quality; then investigate
   embedding caches, parallel retrieval, smaller context, and model selection.

Change one factor at a time. Report functional, semantic, and operational results
separately. Low latency is not success when answers are incorrect or access scope leaks.

## References

- LangSmith privacy: https://docs.langchain.com/langsmith/mask-inputs-outputs
- LangChain tracing: https://docs.langchain.com/langsmith/trace-with-langchain
- Prometheus Python histograms: https://prometheus.github.io/client_python/instrumenting/histogram/
- Prometheus/FastAPI: https://prometheus.github.io/client_python/exporting/http/fastapi-gunicorn/
