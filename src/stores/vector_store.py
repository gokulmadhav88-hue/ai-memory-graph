"""Vector store wrapper for semantic retrieval."""

from __future__ import annotations


class VectorStore:
    """Simple placeholder wrapper around a vector database such as Chroma."""

    def __init__(self, collection_name: str = "memory_graph"):
        self.collection_name = collection_name

    def add(self, documents):
        return {"added": len(documents)}

    def query(self, text: str, top_k: int = 5):
        return {"query": text, "results": []}
