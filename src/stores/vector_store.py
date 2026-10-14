"""Vector store wrapper for semantic retrieval (Person B).

Contract (docs/io_contracts.md):
    add(note_id, text, metadata) -> None      # metadata: doc_id, section, chunk_index, content_hash
    query(query_text, top_k=5) -> list[dict]  # [{"note_id", "score", "text", "metadata"}]
    delete_document(doc_id) -> int            # number of chunks removed
    add_many(records, batch_size=100) -> None # bounded idempotent upsert batches
    delete_notes(note_ids) -> int             # exact old IDs for safe replacement

Scores are normalised to 0-1 (higher = more similar). Chroma returns cosine
distances, converted as score = 1 - distance, clamped to [0, 1].
Raises StoreError on real failures. Empty results are NOT errors.

Usable two ways, so both calling styles work:
    from src.stores import vector_store
    vector_store.add(...)                         # module-level, shared default store
    store = vector_store.VectorStore(...)         # explicit instance (tests, custom config)

Pass `embedding_function=` to select the model before creating a collection. The
runtime supplies the configured embedder; changing models requires a new collection
and reindexing, since vectors from different models cannot be compared reliably.
"""

from __future__ import annotations

import os
from typing import Any
from .errors import StoreError

DEFAULT_COLLECTION = "memory_graph"
DEFAULT_PERSIST_PATH = "data/chroma"   # override with CHROMA_PATH or persist_path=


def _distance_to_score(distance: float) -> float:
    """Cosine distance (0 = identical) -> similarity score in [0, 1]."""
    try:
        score = 1.0 - float(distance)
    except (TypeError, ValueError):
        return 0.0
    if score != score:          # NaN
        return 0.0
    return max(0.0, min(1.0, score))


def _clean_metadata(metadata: Any) -> dict:
    """Chroma metadata must be flat str/int/float/bool and cannot hold None."""
    if not isinstance(metadata, dict):
        raise StoreError("metadata must be a dict")
    clean: dict = {}
    for key, value in metadata.items():
        if value is None:
            continue                                  # e.g. section=None: omitted, read back as missing
        clean[str(key)] = value if isinstance(value, (str, int, float, bool)) else str(value)
    doc_id = clean.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise StoreError("metadata['doc_id'] is required (needed for delete_document)")
    return clean


class VectorStore:
    """Chroma-backed semantic store. Chroma is imported lazily on first use."""

    def __init__(self, collection_name: str = DEFAULT_COLLECTION, persist_path: str | None = None,
                 embedding_function: Any = None, client: Any = None):
        self.collection_name = collection_name
        self._persist_path = persist_path or os.environ.get("CHROMA_PATH") or DEFAULT_PERSIST_PATH
        self._embedding_function = embedding_function
        self._client = client
        self._collection: Any = None

    def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        try:
            if self._client is None:
                if self._embedding_function is None:
                    raise StoreError(
                        "embedding model is a team decision; pass embedding_function before using Chroma"
                    )
                try:
                    import chromadb
                except ImportError as e:
                    raise StoreError("chromadb is not installed (pip install chromadb)") from e
                client = chromadb.PersistentClient(path=self._persist_path)
            else:
                client = self._client
            self._client = client
            kwargs: dict[str, Any] = {"name": self.collection_name,
                                      "metadata": {"hnsw:space": "cosine"}}
            if self._embedding_function is not None:
                kwargs["embedding_function"] = self._embedding_function
            self._collection = client.get_or_create_collection(**kwargs)
        except StoreError:
            raise
        except Exception as e:
            raise StoreError(f"could not open Chroma collection '{self.collection_name}': {e}") from e
        return self._collection

    def add(self, note_id: str, text: str, metadata: dict) -> None:
        self.add_many([{'note_id': note_id, 'text': text, 'metadata': metadata}])

    def add_many(self, records: list[dict], *, batch_size: int = 100) -> None:
        """Validate before writing, then upsert bounded batches by stable note ID.

        A batch can be replayed after a partial failure; Chroma upserts are idempotent.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise StoreError('batch_size must be a positive integer')
        clean = []
        seen = set()
        for record in records:
            if not isinstance(record, dict):
                raise StoreError('vector records must be dicts')
            note_id, text = record.get('note_id'), record.get('text')
            if not isinstance(note_id, str) or not note_id.strip():
                raise StoreError("note_id must be a non-empty string")
            if not isinstance(text, str) or not text.strip():
                raise StoreError("text must be a non-empty string")
            if note_id in seen:
                raise StoreError('A vector batch cannot contain duplicate note IDs')
            seen.add(note_id)
            clean.append((note_id, text, _clean_metadata(record.get('metadata'))))
        if not clean:
            return
        collection = self._get_collection()
        try:
            get_limit = getattr(self._client, 'get_max_batch_size', None)
            if callable(get_limit):
                batch_size = min(batch_size, get_limit())
            for start in range(0, len(clean), batch_size):
                batch = clean[start:start + batch_size]
                collection.upsert(ids=[r[0] for r in batch], documents=[r[1] for r in batch],
                                  metadatas=[r[2] for r in batch])
        except Exception as e:
            raise StoreError(f'vector batch upsert failed: {e}') from e

    def delete_notes(self, note_ids: list[str]) -> int:
        """Delete exact old IDs so cleanup cannot remove a newer document version."""
        if not isinstance(note_ids, (list, tuple)):
            raise StoreError('note_ids must be a list of non-empty strings')
        ids = list(dict.fromkeys(note_ids))
        if not all(isinstance(n, str) and n.strip() for n in ids):
            raise StoreError('note_ids must contain non-empty strings')
        if not ids:
            return 0
        collection = self._get_collection()
        try:
            for start in range(0, len(ids), 100):
                collection.delete(ids=ids[start:start + 100])
        except Exception as e:
            raise StoreError(f'vector note deletion failed: {e}') from e
        return len(ids)

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise StoreError("top_k must be an integer")
        if top_k < 1 or not isinstance(query_text, str) or not query_text.strip():
            return []                                   # nothing sensible to search for
        collection = self._get_collection()
        try:
            count = collection.count()
            if count == 0:
                return []
            res = collection.query(query_texts=[query_text], n_results=min(top_k, count),
                                   include=["documents", "metadatas", "distances"])
        except Exception as e:
            raise StoreError(f"vector query failed: {e}") from e
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        results = []
        for note_id, doc, meta, dist in zip(ids, docs, metas, dists):
            results.append({"note_id": note_id, "score": _distance_to_score(dist),
                            "text": doc or "", "metadata": dict(meta or {})})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def delete_document(self, doc_id: str) -> int:
        if not isinstance(doc_id, str) or not doc_id.strip():
            return 0
        collection = self._get_collection()
        try:
            ids = collection.get(where={"doc_id": doc_id}, include=[]).get("ids") or []
            if ids:
                collection.delete(ids=ids)
        except Exception as e:
            raise StoreError(f"vector delete failed for doc '{doc_id}': {e}") from e
        return len(ids)


# ---------------------------------------------------------------------------
# Module-level API (matches the contract: vector_store.add / query / delete_document)
# ---------------------------------------------------------------------------

_default_store: VectorStore | None = None


def get_default_store() -> VectorStore:
    global _default_store
    if _default_store is None:
        _default_store = VectorStore()
    return _default_store


def set_default_store(store: VectorStore | None) -> None:
    """Swap the shared default (tests, or once the team fixes the embedding config)."""
    global _default_store
    _default_store = store


def add(note_id: str, text: str, metadata: dict) -> None:
    get_default_store().add(note_id, text, metadata)


def add_many(records: list[dict], *, batch_size: int = 100) -> None:
    get_default_store().add_many(records, batch_size=batch_size)


def delete_notes(note_ids: list[str]) -> int:
    return get_default_store().delete_notes(note_ids)


def query(query_text: str, top_k: int = 5) -> list[dict]:
    return get_default_store().query(query_text, top_k)


def delete_document(doc_id: str) -> int:
    return get_default_store().delete_document(doc_id)
