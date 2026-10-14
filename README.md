# AI Memory Graph

Upload documents and answer questions using graph relationships, semantic retrieval, evidence-grounded reasoning, and critic review.

## Overview

The goal is to build a pipeline that ingests documents, extracts structured notes, stores relationships in a graph database, and retrieves evidence through a hybrid graph + vector workflow.

## Repository structure

- `src/` — application source code
- `docs/` — design notes, schema docs, and eval prompts
- `data/` — raw documents and sample extracted notes
- `eval/` — evaluation scripts
- `app/` — UI wrapper
- `tests/` — starter tests

## Current state

The agents now share ingestion and query pipelines, with a transactional Neo4j adapter, persistent Chroma, a Streamlit upload/chat interface, and a separate offline demo.

## Run the app

Run from this README's directory (the inner ai-memory-graph-main folder), with Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app/chatbot_app.py
```

The app starts in **Offline demo**. Click **Load sample document** and ask **Who manages Atlas?**
You can upload TXT, Markdown, PDF, DOCX, CSV, and XLSX. Demo uses scripted extraction,
keyword search, and quoted passages; its persistent data is isolated in data/demo.

## Full pipeline: local Neo4j or Aura

Fill in the LLM API key and Neo4j password in .env (a blank copy has been prepared
locally). On another checkout, copy .env.example to .env first.
All five model agents use the shared client; each can override its provider/model
with QUERY_*, EXTRACTION_*, GRAPH_*, REASONING_*, or CRITIC_*.

For **local Neo4j**, use NEO4J_URI=neo4j://localhost:7687. Start a database in
Neo4j Desktop or, with Docker installed, run:

```powershell
docker compose up -d
```

For **Aura**, set NEO4J_URI=neo4j+s://YOUR_INSTANCE.databases.neo4j.io and use
that database's username and password. Docker is unnecessary.
See [Neo4j connection documentation](https://neo4j.com/docs/python-manual/current/connect/).

Chroma persists in data/chroma and uses local all-MiniLM-L6-v2 embeddings via
DefaultEmbeddingFunction. The first ingestion/search downloads the model. This
v1 choice should be evaluated on your documents. Use a new collection and reingest
when changing embedding models.
See [Chroma embedding documentation](https://docs.trychroma.com/guides/embeddings).

Select **Neo4j + Chroma** and click **Connect workspace**. Upload and ingest documents,
then ask questions. Sources identify the original document, note, section/page when
supplied by the parser, and citation. Changed files sharing a filename stem require
**Replace existing document versions**. Failed extraction preserves the previous
version; replacement commits atomically in the graph. Vector failures remain in
a persistent outbox; use **Retry pending indexing** to recover.

Scanned PDFs require Tesseract when OCR is enabled in parser configuration.
Figure descriptions need a vision callback; the standard UI does not provide one.

## Command line and validation

```powershell
.\.venv\Scripts\python.exe -m src.cli --demo ingest data/raw_documents/demo_atlas.txt
.\.venv\Scripts\python.exe -m src.cli --demo ask "Who manages Atlas?"
.\.venv\Scripts\python.exe -m src.cli check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m eval.run_eval --demo
```

Omit --demo to use configured Neo4j, Chroma, and real models. Evaluation executes
questions and reports responses, insufficiency, and errors. It does not claim
answer-quality accuracy without annotated expected answers.
requirements-lock.txt records the exact dependency versions used for validation;
install it instead of requirements.txt to reproduce that environment.

Live Neo4j validation creates/removes only its own unique test namespace:

```powershell
$env:RUN_NEO4J_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest tests/test_neo4j_adapter.py -q
```

## Integration and limits

Ingestion: parser → extraction → graph memory evolution → Neo4j + vector outbox.
Questions: query plan → graph/vector retrieval → reasoning → critic → cited response.
The critic allows at most two retrieval retries. Drafts failing support checks
become an insufficient-evidence answer; supported partial answers remain visible.

Neo4j commits the complete memory state and graph projection in one transaction.
Browse AMGEntity nodes and AMG_FACT relationships with relation_type, memory_id,
and status. Records are scoped by NEO4J_GRAPH_ID; revision checks reject stale
writers. This v1 loads the corpus in memory and rewrites its projection on commit.
Use one app process with a small corpus; larger deployments need incremental
transactions and database-side traversal.

See [the integration audit](docs/integration_audit.md) for initial gaps, fixes,
remaining validation, and team decisions.
