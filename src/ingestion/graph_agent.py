"""Graph-building agent for deduplication and graph writes."""

from __future__ import annotations


class GraphAgent:
    """Agent 2 placeholder for graph + vector dedup and write operations."""

    def __init__(self, graph_store=None, vector_store=None):
        self.graph_store = graph_store
        self.vector_store = vector_store

    def deduplicate(self, notes):
        return notes

    def write(self, notes):
        return {"written": len(notes)}
