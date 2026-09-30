"""Reasoning pipeline components."""
"""reasoning package: Reasoning Agent (and, later, Critic Agent).

Public entry points used by src/pipeline.py:
    from reasoning import write_answer
When critic_agent.py exists, add:
    from .critic_agent import review_answer
"""

from .reasoning_agent import ReasoningConfig, write_answer
from .schemas import AnswerDraft, AnswerState, ClaimSource, NormalizedEvidence, SchemaError

__all__ = [
    "write_answer", "ReasoningConfig",
    "AnswerDraft", "AnswerState", "ClaimSource", "NormalizedEvidence", "SchemaError",
]