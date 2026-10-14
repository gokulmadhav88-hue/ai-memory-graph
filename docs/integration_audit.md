# Integration audit — 14 October 2026

## What was disconnected

| Finding | Consequence | Resolution |
|---|---|---|
| Local query/retrieval/vector files and tests were scaffold versions relative to f0341ab | No working query planning, evidence bundles, or vector persistence | Preserved local files in .integration-backup/before-integration; recovered implemented files from Git HEAD and adapted them |
| Neo4jStore returned success without connecting | No actual database writes; GraphAgent rejected missing capabilities | Added driver connectivity, constraints, transactional persistence, graph projection, refresh, and revision checks |
| GraphAgent used mappings/snapshots; retrieval required query_subgraph | Independent components could not share a store | Both adapters now expose the same memory and retrieval interface |
| No ingestion orchestration | Parsing never reached extraction/storage in one entry point | Added reports, duplicate/version checks, atomic graph replacement, partial retry, and vector outbox reporting |
| No query orchestration | Query, retrieval, reasoning, and critic were disconnected | Added history, refined hybrid retries, citations, and fallback |
| Critic was a placeholder locally and on GitHub main | Evidence count substituted for answer review; no retry cap | Added structural/value checks, model support audit, two-retry cap, and rejection after the cap |
| Retrieval ignored memory status | Superseded graph facts and stale vectors could return as current evidence | Traversal selects active memories; vector hits resolve note status in graph |
| Missing runtime dependencies, embeddings, and DB environment | Full runtime could not open stores or call providers | Added dependencies, shared errors, explicit local embedding choice, environment configuration, and Docker Compose |
| UI/evaluation were placeholders | No upload/chat interface or actual evaluation runs | Added Streamlit, headless CLI, evaluation execution, and isolated offline demo |
| Documents described old contracts | Team members had conflicting integration targets | Updated README and contract addendum |

The critic-related commit dc698cd changed only README and .env.example; its
message did not correspond to a committed critic implementation. Only main was
found on the connected GitHub repository during this audit.

## Verification

The offline document-to-answer smoke test passes, including persistence and
citations. Tests cover replacement/rollback, duplicates, partial failures,
index recovery, active-status filtering, shared provenance, critic rejection,
retry limits, adapter transactions with a fake driver, and Streamlit AppTest.
Real Chroma tests use a deterministic embedder without downloading a model.

Exact suite results are recorded below after validation. Fake-driver tests do
not validate live Cypher or server compatibility.

Final local validation: **379 passed, 4 skipped** with Python 3.11. The skips are
one missing large-file fixture, two optional Tesseract OCR checks, and the opt-in
live Neo4j test. Eleven deprecation warnings originate from legacy test embedding
functions, not the runtime's DefaultEmbeddingFunction. Changed-file Ruff checks,
Python compilation, and git diff --check pass. pip check reports no broken requirements.
The dependency versions used are saved in requirements-lock.txt.

## Required before calling the full version finished

1. Configure .env with local Neo4j or Aura and a valid model API key. Run the live
   adapter test and a real document/query smoke test. No database or model
   credentials were available during this audit.
2. Evaluate retrieval/answer quality on your annotated documents and questions.
   Tune similarity threshold and top_k. The evaluation runner has no accuracy
   ground truth yet.
3. Agree on identity/update semantics: filename stem is the document ID;
   predicates are free-form snake_case; only managed_by/led_by are single-valued
   by default. Duplicate stems with changed content require explicit replacement.
4. Validate the local embedding choice for your languages and document types.
   Use a new collection and reingest when changing models.
5. Decide whether chat conversations should become stored knowledge. Extraction
   supports conversation notes; UI history supplies question context only.

## First-version limits

- Small corpus, one app process. Neo4j snapshot/projection rewrites prioritize
  compatibility with GraphAgent over scale.
- Neo4j/Chroma do not share a transaction. Pending vector upserts retry; graph
  status excludes stale hits. Failed vector deletions can leave unusable entries.
- Supersession operates on whole notes. Mixed predicates are retained conservatively.
- Source section/page is available only where the parser supplies it.
- OCR requires Tesseract; figures need a vision callback and are skipped by this UI.
- External web reasoning remains opt-in and is not enabled in the document UI.
- Authentication, user isolation, and deployment are outside this local team version.
- Local changes have not been committed, pushed, or merged on GitHub.
