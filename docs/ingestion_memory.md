# Ingestion memory evolution contract (v1)

Implemented in `extraction_agent.py` and `graph_agent.py`. This supplements the older I/O contract; downstream retrieval and Neo4j adapters need updating before full integration.

## Extraction

Inject an object providing `complete_json(prompt, system=..., max_tokens=...)`. The shared `src/llm_client.py` is currently a placeholder and is not silently replaced with fake extraction.

`ExtractionAgent(llm).extract_note(chunk, source_type="document", source_time=None)` returns `ExtractedNote`, convertible with `to_dict()`. `extract_conversation(turns, conversation_id)` accepts user/assistant turns with content and optional timezone-aware timestamp. Conversation text includes the speaker; treat claims as assertions, not verified facts. Document and conversation IDs must be namespaced by callers to prevent cross-user collisions. Conversation storage must be explicitly invoked by the application, not automatically enabled for every chat.

Notes preserve doc_id, chunk_index, content_hash, verbatim content, section, content_type, source_filename, page, source_type, source_time, created_at. IDs are deterministic hashes of source identity, text, and source time. Relations additionally require an exact evidence_quote and optional event_time. Source time is never used as event time. Quotes establish traceability, not logical entailment; model extraction and identity resolution still require evaluation.

Validation retries extraction once, rejects invented quote strings, invalid entity types, malformed relation endpoints and timestamps. Uninformative notes carry empty extraction fields. Secret exclusion is prompted, not a complete DLP guarantee. Do not ingest sensitive credentials.

## Graph and memory

`GraphAgent(store, vector_store=None, llm=None).ingest_note(note)` returns the standard node/edge ingestion result plus memory_id and decision. Decisions contain action, target_ids, reason, and optionally blocked_proposal. Actions: add, merge, update, retain, retire.

Without an LLM, exact duplicate text merges and distinct evidence is added. With an LLM, ambiguous entity resolution and memory decisions are proposed. Code verifies target IDs, entity types, equivalent relation sets before consolidation, and explicit later event times with matching subject/predicate sets before updates. Unknown or ambiguous facts are preserved. Retirement is reversible through history; no source records are deleted. Retain leaves previous memories untouched and stores incoming evidence separately. This first version evolves whole notes: mixed-fact notes are conservatively retained when predicates differ. It does not provide arbitrary fact-level edits, semantic contradiction guarantees, or usage-based pruning.

`MemoryStore()` is in-memory. `MemoryStore(path)` persists nodes, edges, original notes, active/superseded/retired memories, revisions, and audit events to an atomically replaced JSON file. Intended for single-process development. This is a working local reference adapter, not a Neo4j connection. A production Neo4j adapter must implement the lock/snapshot/restore/commit and record mappings used by GraphAgent, preferably replacing snapshot operations with database transactions. Existing placeholder Neo4jStore is rejected explicitly.

Each memory links its source_note_ids. Edges reference memory_id; consumers must join to memory status before treating an edge as current. Original edges remain for historical retrieval. Multi-target merges retire redundant memories and preserve their source references. Events preserve decisions; revisions preserve previous state.

Vector indexing uses a persistent outbox flag on notes. `flush_vector_outbox()` retries failed writes and returns errors. Vector `add` MUST upsert by note_id. Retrieval MUST resolve each vector hit's memory_id against current graph status: cached vector metadata alone is not authoritative. Failed index writes do not discard committed knowledge. Pipeline callers should expose pending index writes to users. Two-store commits are not distributed transactions.

## Example

```python
from src.ingestion.extraction_agent import ExtractionAgent
from src.ingestion.graph_agent import GraphAgent, MemoryStore

extractor = ExtractionAgent(llm=your_client)
graph = GraphAgent(MemoryStore("data/memory_store.json"), llm=your_client)
for chunk in parsed_chunks:
    report = graph.ingest_note(extractor.extract_note(chunk))
```

## Integration remaining

Implement the real LLM client, Neo4j adapter, vector upsert adapter, ingestion pipeline, and downstream active/history filtering. Agree on entity/relation vocabularies and temporal update semantics with teammates. A more recent timestamp alone cannot establish trust or resolve arbitrary multi-valued relationships.

Updates are restricted to configured single-valued predicates (exclusive_relations; defaults managed_by and led_by). Configure this with the team; multi-valued facts such as invested_in must accumulate rather than overwrite.
