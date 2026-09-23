# Data Contract

## Overview

This project expects three core artifact types:

1. Raw source documents
2. Extracted notes
3. Graph and vector indexes

## Note Schema

Each extracted note should include:

- `note_id`: stable identifier
- `source_document_id`: source doc identifier
- `title`: short title or summary
- `content`: the extracted text
- `entities`: list of named entities
- `keywords`: relevant terms or key phrases
- `tags`: thematic tags
- `source_span`: start/end offsets or chunk references
- `created_at`: timestamp

## Node Schema

Graph nodes should be normalized to:

- `Entity` node
  - `id`
  - `name`
  - `type`
  - `aliases`

- `Concept` node
  - `id`
  - `name`
  - `category`

- `Document` node
  - `id`
  - `title`
  - `source_path`

## Edge Schema

Graph edges should capture relationships:

- `MENTIONS`
- `RELATED_TO`
- `HAS_TAG`
- `REFERENCES`
- `SAME_AS`

## Validation

All extracted notes and graph writes should enforce:

- deduplicated node names where possible
- consistent entity types
- source traceability
- explicit confidence or evidence references when available
