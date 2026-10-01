# I/O Contracts: AI Memory Graph (v2)

The single source of truth for what every file takes in and gives back.
**If you change a shape here, check the dependency map (section 5), update the code, and add a line to the edit log (section 7).** Announce it to the team before you push.

## 0. Ownership (proposed, confirm as a team)

| Area | Files | Owner |
|---|---|---|
| Document parser | `src/ingestion/parser/*` | Person C (then handed to A as a finished function) |
| Extraction + graph writing | `extraction_agent.py`, `graph_agent.py`, `src/ingestion/pipeline.py` (new) | Person A |
| Graph store | `stores/neo4j_store.py` | Person A |
| Query + retrieval | `query_agent.py`, `retrieval_agent.py` | Person B |
| Vector store | `stores/vector_store.py` | Person B |
| Reasoning + critic | `reasoning_agent.py`, `critic_agent.py` | Person C |
| Shared LLM wrapper | `src/llm_client.py` | Person C (write it first, everything depends on it) |
| Query orchestration | `src/pipeline.py` (new) | Person C |
| Eval + UI | `eval/run_eval.py`, `app/chatbot_app.py` | Person C |

Two files are new compared with the original structure: `src/ingestion/pipeline.py` and `src/pipeline.py`. Nothing was responsible for chaining the agents, and `run_eval.py` needs to call the pipeline without going through Streamlit.

## 1. Conventions

- Python 3.14. Shapes are plain `dict`s (or dataclasses that convert with `.to_dict()`). Everything must be JSON-serializable.
- All ids are `str`. Timestamps are ISO 8601 UTC strings. Scores are `float` in 0 to 1, higher means better.
- **Empty results are not errors.** Return `[]`, an empty bundle, and so on. Raise a typed error only for real failures.
- **Chunk text and evidence text are DATA, never instructions.** Every prompt that includes them must say so (prompt-injection defense).
- Every agent calls the LLM only through `llm_client.py`.
- Every agent must be testable with a fake `llm_client` and fake stores.

**Errors** (all subclass `Exception`): `ParserError` (and `UnsupportedFileType`, `FileTooLarge`, `EncryptedFile`, `InvalidFile`), `ExtractionError`, `LLMError`, `StoreError`.

**Allowed entity types:** `Person`, `Organization`, `Product`, `Location`, `Event`, `Concept`, `Date`, `Other`. Keep the list small and agreed.

## 2. Shared shapes

### Chunk (parser output, defined in code in `parser/schema.py`)
```python
Chunk = {
    "doc_id": str,               # filename stem for now (provisional, see open decisions)
    "chunk_index": int,          # 0-based, document order
    "text": str,                 # cleaned text; may contain "\n\n" where paragraphs merged;
                                 # may start with 1 overlap sentence copied from the previous chunk
    "source_filename": str,
    "content_hash": str,         # SHA-256 of the whole source file
    "parser_version": str,
    "content_type": str,         # "text" | "table" | "image_description" (only "text" for now)
    "page": int | None,
    "section": str | None,       # nearest heading, if any
}
```

### Note (extraction_agent output)
```python
Entity   = {"name": str, "type": str}                         # type from the allowed list
Relation = {"subject": str, "relation": str, "object": str}   # subject/object must match an Entity name

Note = {
    "note_id": str,              # uuid4 hex, created by extraction_agent
    "doc_id": str,               # = Chunk.doc_id
    "chunk_index": int,          # = Chunk.chunk_index
    "content_hash": str,         # = Chunk.content_hash
    "section": str | None,
    "content_type": str,
    "content": str,              # the chunk text, verbatim (this is what gets embedded)
    "is_informative": bool,      # False = nothing worth storing (noise layer 2)
    "entities": list[Entity],
    "relations": list[Relation],
    "keywords": list[str],
    "tags": list[str],
    "context": str,              # 1-2 sentence summary of what this note is about
    "created_at": str,
}
```
Rules: use the fullest name as written ("OpenAI", not "the company"), resolving pronouns from the chunk. `relation` is lowercase snake_case, subject to object (`released`, `co_founded_by`, `ceo_of`, `invested_in`, `based_in`). If `is_informative` is `False`, then `entities`, `relations`, `keywords`, `tags` are `[]` and `context` is `""`.

### Graph shapes (graph_agent to neo4j_store)
```python
GraphNode = {
    "node_id": str,              # stable; reused when an entity already exists
    "name": str,                 # canonical name
    "entity_type": str,
    "aliases": list[str],        # e.g. ["OpenAI Inc."] after a merge
    "source_note_ids": list[str],
    "properties": dict,
}
GraphEdge = {
    "edge_id": str,
    "source_node_id": str,
    "target_node_id": str,
    "relation_type": str,        # neo4j_store uppercases it for the database
    "source_note_ids": list[str],
}
Subgraph = {"nodes": list[GraphNode], "edges": list[GraphEdge]}
```

### Ingestion results
```python
IngestResult = {                 # one Note
    "note_id": str, "skipped": bool, "skip_reason": str | None,
    "nodes_created": list[str],
    "nodes_merged": list[{"entity": str, "into_node_id": str}],
    "edges_created": list[str],
}
IngestReport = {                 # one document
    "doc_id": str, "content_hash": str,
    "status": "ingested" | "skipped_duplicate" | "replaced" | "failed",
    "chunks_total": int, "notes_informative": int, "notes_skipped": int,
    "nodes_created": int, "nodes_merged": int, "edges_created": int,
    "errors": list[str],
}
```

### Query-side shapes
```python
RetrievalPlan = {
    "original_question": str,
    "rewritten_query": str,             # clarified, self-contained version
    "route": "graph" | "vector" | "hybrid",
    "target_entities": list[str],       # entities the question is about, if any
    "attempt": int,                     # 0 first try, 1 and 2 are retries
}

EvidenceChunk = {"evidence_id": str,    # "E1", "E2", ...
                 "note_id": str, "doc_id": str, "section": str | None,
                 "text": str, "score": float, "source": "vector" | "graph"}
GraphFact     = {"evidence_id": str,    # "F1", "F2", ...
                 "subject": str, "relation": str, "object": str,
                 "source_note_ids": list[str]}
EvidenceBundle = {
    "chunks": list[EvidenceChunk],
    "graph_facts": list[GraphFact],
    "was_filtered": bool,               # True once noise filtering ran
    "route_used": str,
    "notes": str | None,                # e.g. "graph returned nothing, used vector only"
}

AnswerDraft = {
    "answer_text": str,
    "evidence_used": list[str],         # evidence_ids actually relied on
    "confidence": float,                # 0-1, self-reported
    "evidence_sufficient": bool,
    "missing_info": str | None,         # what was missing, helps the critic refine
}

CriticDecision = {
    "action": "accept" | "retry",
    "refined_query": str | None,        # set only when action == "retry"
    "reason": str,
    "retry_count": int,                 # retries already used before this review
}

FinalResponse = {
    "answer_text": str,
    "sources": list[{"doc_id": str, "section": str | None, "note_id": str}],
    "confidence": float,
    "retries_used": int,
    "insufficient_evidence": bool,
}
```
Fallback answer when evidence never becomes sufficient: `"I don't have enough information in the uploaded documents to answer that."`

## 3. Function contracts

### Ingestion side

**`parser.parse_document(file_path: str | Path) -> list[Chunk]`** (Person C)
Raises `UnsupportedFileType`, `FileTooLarge`, `EncryptedFile`, `InvalidFile`. Returns `[]` for an empty file. Deterministic, no LLM. Import: `from src.ingestion.parser import parse_document, Chunk, ParserError`.

**`extraction_agent.extract_note(chunk: Chunk) -> Note`** (Person A)
Calls `llm_client.complete_json`. Retries once on bad JSON, then raises `ExtractionError`. Must set `is_informative=False` for noise instead of inventing entities. Prompt includes the "chunk text is data" rule.

**`graph_agent.ingest_note(note: Note) -> IngestResult`** (Person A)
Skips notes with `is_informative=False`. For each entity: ask `neo4j_store.find_similar_node`, then exact match, then LLM tie-break for ambiguous cases, then reuse or create the node. Writes edges. Calls `vector_store.add(note_id, content, metadata)`. Calls `llm_client` for the dedup judgment.

**`src/ingestion/pipeline.ingest_document(file_path, replace: bool = False) -> IngestReport`** (Person A)
1. `parse_document`. 2. If `neo4j_store.document_exists(doc_id, content_hash)` then status `skipped_duplicate`. 3. If the same `doc_id` exists with a different hash, delete the old data (`delete_document` on both stores) and status `replaced`. 4. For each chunk: `extract_note`, then `ingest_note`. One bad chunk is logged into `errors` and never stops the rest.

### Stores

**`neo4j_store`** (Person A)
```python
write_node(node: GraphNode) -> str                          # returns node_id (upsert)
write_edge(edge: GraphEdge) -> str                          # returns edge_id
get_node(node_id: str) -> GraphNode | None
find_similar_node(name: str, entity_type: str | None = None, limit: int = 5) -> list[GraphNode]
query_subgraph(entity_names: list[str], depth: int = 2) -> Subgraph
document_exists(doc_id: str, content_hash: str) -> bool
delete_document(doc_id: str) -> int                         # returns nodes/edges removed
```
Raises `StoreError` on connection failure.

**`vector_store`** (Person B)
```python
add(note_id: str, text: str, metadata: dict) -> None       # metadata: doc_id, section, chunk_index, content_hash
query(query_text: str, top_k: int = 5) -> list[dict]        # [{"note_id","score","text","metadata"}]
delete_document(doc_id: str) -> int
```

**`llm_client`** (Person C)
```python
complete(prompt: str, system: str | None = None, max_tokens: int = 1000, temperature: float = 0.0) -> str
complete_json(prompt: str, system: str | None = None, max_tokens: int = 1000) -> dict   # parses, retries once, else LLMError
```
Provide a fake mode (for example `LLM_FAKE=1`) that returns canned outputs, so everyone can develop and test without API cost.

### Query side

**`query_agent.plan_query(question: str, history: list[dict] | None = None) -> RetrievalPlan`** (Person B)
Rewrites the question, spots entities, picks the route: relationship or multi-hop questions go `graph`, fuzzy or topic questions go `vector`, mixed go `hybrid`. `attempt` is 0.

**`retrieval_agent.retrieve(plan: RetrievalPlan, top_k: int = 5) -> EvidenceBundle`** (Person B)
Reads from `neo4j_store` and `vector_store` per the route, merges, filters noise, assigns `evidence_id`s. On a retry, the caller passes a plan whose `rewritten_query` is the critic's `refined_query` and whose `attempt` is incremented.

**`reasoning_agent.write_answer(question: str, evidence: EvidenceBundle) -> AnswerDraft`** (Person C)
Answers only from the evidence. If the evidence doesn't contain the answer, set `evidence_sufficient=False` and fill `missing_info`. Cites `evidence_id`s.

**`critic_agent.review_answer(question, draft: AnswerDraft, evidence: EvidenceBundle, retry_count: int) -> CriticDecision`** (Person C)
Hard rule: if `retry_count >= 2`, always return `action="accept"`. The cap lives here and nowhere else.

**`src/pipeline.answer_question(question: str, history: list[dict] | None = None) -> FinalResponse`** (Person C)
```
plan = plan_query(question)
retries = 0
loop:
    evidence = retrieve(plan)
    draft    = write_answer(question, evidence)
    decision = review_answer(question, draft, evidence, retries)
    if decision.action == "accept": break
    plan = {**plan, "rewritten_query": decision.refined_query, "attempt": retries + 1}
    retries += 1
build FinalResponse (fallback text if evidence never became sufficient)
```

## 4. Worked example (use these as fakes in `data/sample_notes/`)

Document `ai_news.txt` (the three paragraphs merge into one chunk, since each is under 30 words):

```json
{"doc_id": "ai_news", "chunk_index": 0, "source_filename": "ai_news.txt",
 "text": "OpenAI released GPT-4 in March 2023. The company was co-founded by Sam Altman.\n\nSam Altman is the CEO of OpenAI. Microsoft has invested billions in the company.\n\nOpenAI Inc. is based in San Francisco.",
 "content_hash": "abc123", "parser_version": "0.2.0", "content_type": "text", "page": null, "section": null}
```

The resulting Note:
```json
{"note_id": "n1", "doc_id": "ai_news", "chunk_index": 0, "content_hash": "abc123",
 "section": null, "content_type": "text",
 "content": "<same text as the chunk>",
 "is_informative": true,
 "entities": [{"name": "OpenAI", "type": "Organization"}, {"name": "GPT-4", "type": "Product"},
              {"name": "Sam Altman", "type": "Person"}, {"name": "Microsoft", "type": "Organization"},
              {"name": "San Francisco", "type": "Location"}],
 "relations": [{"subject": "OpenAI", "relation": "released", "object": "GPT-4"},
               {"subject": "OpenAI", "relation": "co_founded_by", "object": "Sam Altman"},
               {"subject": "Sam Altman", "relation": "ceo_of", "object": "OpenAI"},
               {"subject": "Microsoft", "relation": "invested_in", "object": "OpenAI"},
               {"subject": "OpenAI", "relation": "based_in", "object": "San Francisco"}],
 "keywords": ["GPT-4", "CEO", "investment"], "tags": ["AI", "company"],
 "context": "Describes OpenAI's GPT-4 release, its leadership, funding, and location.",
 "created_at": "2026-09-28T00:00:00Z"}
```
Note: "OpenAI Inc." in the text is merged into the "OpenAI" node by the graph agent, and "OpenAI Inc." becomes an alias.

Question: "Who leads the company that made GPT-4?"
```json
{"original_question": "Who leads the company that made GPT-4?",
 "rewritten_query": "Who is the CEO of the company that released GPT-4?",
 "route": "graph", "target_entities": ["GPT-4"], "attempt": 0}
```
Evidence:
```json
{"chunks": [{"evidence_id": "E1", "note_id": "n1", "doc_id": "ai_news", "section": null,
             "text": "Sam Altman is the CEO of OpenAI...", "score": 0.82, "source": "vector"}],
 "graph_facts": [{"evidence_id": "F1", "subject": "OpenAI", "relation": "released", "object": "GPT-4", "source_note_ids": ["n1"]},
                 {"evidence_id": "F2", "subject": "Sam Altman", "relation": "ceo_of", "object": "OpenAI", "source_note_ids": ["n1"]}],
 "was_filtered": true, "route_used": "graph", "notes": null}
```
Draft, decision, final:
```json
{"answer_text": "Sam Altman is the CEO of OpenAI, the company that released GPT-4.",
 "evidence_used": ["F1", "F2"], "confidence": 0.9, "evidence_sufficient": true, "missing_info": null}
{"action": "accept", "refined_query": null, "reason": "Answer is supported by F1 and F2.", "retry_count": 0}
{"answer_text": "Sam Altman is the CEO of OpenAI, the company that released GPT-4.",
 "sources": [{"doc_id": "ai_news", "section": null, "note_id": "n1"}],
 "confidence": 0.9, "retries_used": 0, "insufficient_evidence": false}
```

## 5. Dependency map (who breaks if you change a shape)

| Shape | Produced by | Consumed by |
|---|---|---|
| `Chunk` | parser | extraction_agent, ingestion/pipeline |
| `Note` | extraction_agent | graph_agent, ingestion/pipeline, tests |
| `GraphNode` / `GraphEdge` / `Subgraph` | graph_agent, neo4j_store | neo4j_store, retrieval_agent |
| vector `query` result | vector_store | retrieval_agent |
| `RetrievalPlan` | query_agent, pipeline (retries) | retrieval_agent |
| `EvidenceBundle` | retrieval_agent | reasoning_agent, critic_agent, pipeline |
| `AnswerDraft` | reasoning_agent | critic_agent, pipeline |
| `CriticDecision` | critic_agent | pipeline |
| `FinalResponse` | pipeline | chatbot_app, run_eval |
| `llm_client` return types | llm_client | every agent |

## 6. Open decisions (settle as a team, then delete the line and log it)

1. **`doc_id`:** filename stem (current) or content hash? They differ when a file is renamed or edited.
2. **Relation vocabulary:** free-form snake_case (current) or a fixed list? Free-form gives a messier graph, and a fixed list can miss facts.
3. **Dedup rules:** the name-similarity threshold, and when to call the LLM tie-break.
4. **Embedding model** for the vector store.
5. **Retrieval defaults:** `top_k`, and how graph and vector results are merged (rank fusion or sequential).
6. **Confidence:** is the self-reported number used by the critic, or ignored? (Self-reported confidence is unreliable, so the critic should mainly use `evidence_sufficient`.)
7. **`history`:** is session memory in scope for v1? The parameter exists but is optional.

## 7. Edit log

| Date | Change | By | Downstream updated? |
|---|---|---|---|
| 2026-09-28 | `Chunk` gains `content_type`, `content_hash`, `section`, `parser_version`; `source_filename` and `page` kept | C | extraction_agent (pending) |
| 2026-09-28 | `Note.source_doc` renamed `doc_id`; `Note.entities` is now `list[{name, type}]` (was `list[str]`); added `is_informative`, `content_hash`, `section`, `content_type` | C | graph_agent (pending) |
| 2026-09-28 | `GraphNode.entity_name` renamed `name`; added `aliases` | C | neo4j_store (pending) |
| 2026-09-28 | `find_similar_node` now returns `list[GraphNode]` (was a single node or None) | C | graph_agent (pending) |
| 2026-09-28 | Added `evidence_id` to evidence items, `missing_info` to `AnswerDraft`, `attempt` to `RetrievalPlan`, `FinalResponse` | C | B, C (pending) |
| 2026-09-28 | New files: `src/ingestion/pipeline.py`, `src/pipeline.py`; parser ownership moved to C | C | architecture.md (pending) |
