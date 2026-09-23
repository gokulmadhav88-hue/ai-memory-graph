"""Shared LLM client wrapper."""

from __future__ import annotations

from typing import Any


class LLMClient:
    """Thin wrapper around the LLM provider."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to the underlying LLM provider.

        This is intentionally a placeholder for the real implementation.
        """
        raise NotImplementedError("Connect this wrapper to your actual LLM provider.")
