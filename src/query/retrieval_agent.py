"""Retrieval agent for combining evidence sources."""

from __future__ import annotations


class RetrievalAgent:
    """Agent 4 placeholder for graph/vector fusion and filtering."""

    def retrieve(self, question: str, graph_results=None, vector_results=None):
        return {
            "question": question,
            "graph_results": graph_results or [],
            "vector_results": vector_results or [],
        }
