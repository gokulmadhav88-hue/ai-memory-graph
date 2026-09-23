"""Reasoning agent for producing final answers."""

from __future__ import annotations


class ReasoningAgent:
    """Agent 5 placeholder for answer synthesis from evidence."""

    def answer(self, question: str, evidence: list[dict] | None = None) -> str:
        evidence = evidence or []
        if not evidence:
            return f"No evidence was found for: {question}"
        return f"Based on {len(evidence)} evidence items, the answer is: {question}"
