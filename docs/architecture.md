# Architecture

## High-level view

The system is organized into a pipeline:

1. Ingest raw documents
2. Extract structured notes
3. Normalize and deduplicate into a graph + vector store
4. Answer queries via graph and semantic retrieval
5. Reason and critique the answer before returning it

## Agent roles

- Ingestion: parses text and extracts structured notes
- Graph building: deduplicates and writes graph evidence
- Querying: chooses the retrieval route
- Retrieval: merges graph and vector evidence
- Reasoning: creates the final answer
- Critic: judges sufficiency and retries when needed

## Core vs stretch

### Core

- document parsing
- note extraction
- graph storage in Neo4j
- vector retrieval in Chroma
- query + retrieval pipeline
- reasoning and critique loop
- basic eval harness

### Stretch

- multi-hop reasoning across graph paths
- active learning or note refinement
- human-in-the-loop QA review
- async ingestion workers
- production observability and tracing
- advanced reranking and memory pruning
