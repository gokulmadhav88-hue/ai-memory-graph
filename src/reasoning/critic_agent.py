"""Critic agent for judging evidence sufficiency."""

from __future__ import annotations


class CriticAgent:
    """Agent 6 placeholder for evidence review and retry triggers."""

    def assess(self, question: str, evidence: list[dict] | None = None) -> dict:
        items = evidence or []
        return {
            "question": question,
            "sufficient": len(items) > 0,
            "evidence_count": len(items),
            "retry": len(items) == 0,
        }
