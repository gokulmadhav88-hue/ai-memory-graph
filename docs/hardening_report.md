# Reliability and performance changes — 14 October 2026

| Area | Concrete improvement | Remaining tradeoff |
|---|---|---|
| Ingestion | Commit a document graph once, batch vectors and acknowledge once. | Rollback snapshots still grow with corpus size; new ingestion can be reported partial. |
| Neo4j | Skip unchanged commits; project only changed records; retain revision checks. | Full state payload still serializes per changed commit; traversal loads it in Python. |
| Vectors | Batched upserts, durable exact-ID replacement/supersession cleanup, active filtering during failures. | Index may lag; no distributed transaction/background worker. |
| Read concurrency | Parsing/extraction/planning/reasoning/critic release graph locks; retrieval captures sources consistently. | Graph integration decisions/index acknowledgements serialize; citations represent retrieval-time evidence. |
| LLM | Reusable SDK connections, explicit timeout/retry bounds, cleanup, sanitized errors, thread-local usage. | Timeout is per attempt, not a complete-question deadline; live model compatibility needs testing. |
| Query | Bounded input/history/retries, reused plans, configurable result/depth/score budgets. | Limits can omit evidence; no learned reranking or perfect recall. |
| Critic | Bounded audit includes uncited retrieved facts for conflicts; suppressed state/sources remain consistent. | Model review cannot prove correctness. |
| Parser | Page/OCR provenance, mixed PDFs, short-fact handling, archive/row/overlap bounds. | Structure/OCR quality, truncation, and vision callbacks remain limits. |
| Runtime/UI | Cleanup, shared demo store, fail-closed team OIDC/allowlist, bounded uploads/history, timings/usage. | One shared team corpus; no public tenant isolation. |
| Evaluation/deploy | Annotated checks, failure-aware denominator, latency, dependency lock, CI/container artifacts. | Heuristics do not prove semantic accuracy; hosted artifacts remain untested. |

## Measured operations

The deterministic 12-chunk regression asserts **2 graph commits and 1 vector
batch**, previously **25 commits and 12 individual vector writes**. This measures
operation reduction, not equivalent wall-clock speedup on live Neo4j.

The projection regression verifies that vector acknowledgements change only note
rows, and an unchanged commit sends no database transaction.

## Validation

Full local suite: **467 passed, 4 skipped**, Python 3.11. Skips: opt-in live Neo4j,
two Tesseract checks, and missing large-file fixture. Thirteen warnings come from
legacy deterministic Chroma test embedders. All changed Python files pass Ruff and
pip check passes. Repository-wide style has pre-existing warnings.

Tests use real persistent Chroma with a deterministic embedder, Streamlit AppTest,
and a fake Neo4j driver. Default model weights, live Neo4j/models, OIDC redirects
and claims, Docker build, Linux CI and hosted load still need deployment validation.
