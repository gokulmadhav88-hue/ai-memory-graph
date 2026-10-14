"""Shared LLM client wrapper (Person C).

One place that knows how to reach a model. Agents never import a provider SDK.

* Different agents can use DIFFERENT models: `for_agent("reasoning")` reads
  REASONING_MODEL / REASONING_PROVIDER, falling back to LLM_MODEL / LLM_PROVIDER.
  Some agents may not call the LLM at all.
* Contract functions (docs/io_contracts.md):
      complete(prompt, system=None, max_tokens=1000, temperature=0.0) -> str
      complete_json(prompt, system=None, max_tokens=1000) -> dict   # parses, retries once, else LLMError
* Fake mode: set LLM_FAKE=1 and no network or key is needed.
* API keys come from the environment only (OPENAI_API_KEY, ANTHROPIC_API_KEY).
  Never put a key in code or in .env.example.

Usage in the pipeline:
    reasoning_llm = llm_client.for_agent("reasoning")
    draft = write_answer(question, evidence, llm=reasoning_llm)

NOTE: the real provider adapters (`_openai`, `_anthropic`) have not been run
against a live API yet. Run a smoke test once a key is available.
"""

from __future__ import annotations

import json
import math
import os
import re
from threading import RLock, local
from time import perf_counter
from typing import Any, Callable


class LLMError(Exception):
    """Any failure to get a usable answer from the model."""


# provider function signature:
#   fn(model, prompt, system, max_tokens, temperature, api_key, json_mode) -> str
ProviderFn = Callable[..., str]

DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Configuration (environment only)
# ---------------------------------------------------------------------------

def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def _fake_enabled() -> bool:
    return (_env("LLM_FAKE") or "0").lower() in ("1", "true", "yes", "on")


def _resolve(agent: str | None, provider: str | None, model: str | None) -> tuple[str, str]:
    """Pick (provider, model): explicit argument > AGENT_* env > LLM_* env > defaults."""
    prefix = agent.upper() if agent else None
    if _fake_enabled():
        return "fake", model or "fake"
    provider = (provider
                or (_env(f"{prefix}_PROVIDER") if prefix else None)
                or _env("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    model = (model
             or (_env(f"{prefix}_MODEL") if prefix else None)
             or _env("LLM_MODEL"))
    if model is None and provider == "openai":          # legacy variable from the old .env.example
        model = _env("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    if model is None and provider == "fake":
        model = "fake"
    if model is None:
        where = f"{prefix}_MODEL or " if prefix else ""
        raise LLMError(f"no model configured for provider '{provider}': set {where}LLM_MODEL")
    return provider, model


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _parse_json_object(text: str) -> dict:
    """Parse a JSON object from model text. Tolerates ```json fences and short
    prose around the object. Raises ValueError if no JSON object is found."""
    cleaned = _FENCE.sub("", text.strip()).strip()
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.append(cleaned[start:end + 1])
    last: Exception | None = None
    for c in candidates:
        try:
            value = json.loads(c)
        except json.JSONDecodeError as e:
            last = e
            continue
        if isinstance(value, dict):
            return value
        last = ValueError(f"expected a JSON object, got {type(value).__name__}")
    raise ValueError(str(last) if last else "no JSON found")


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

_FAKE_HANDLER: Callable[[str, str | None, bool], str] | None = None


def set_fake_handler(handler: Callable[[str, str | None, bool], str] | None) -> None:
    """Optional: script the fake provider. handler(prompt, system, json_mode) -> str."""
    global _FAKE_HANDLER
    _FAKE_HANDLER = handler


def _fake(model, prompt, system, max_tokens, temperature, api_key, json_mode) -> str:
    if _FAKE_HANDLER is not None:
        return _FAKE_HANDLER(prompt, system, json_mode)
    return '{"fake": true}' if json_mode else f"[fake] {prompt[:60]}"


def _openai(model, prompt, system, max_tokens, temperature, api_key, json_mode) -> str:
    api_key = api_key or _env("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("openai is not configured: set OPENAI_API_KEY")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise LLMError("the 'openai' package is not installed (pip install openai)") from e
    with OpenAI(api_key=api_key, timeout=60.0, max_retries=1) as client:
        return _openai_request(client, model, prompt, system, max_tokens, temperature, json_mode)[0]


def _openai_request(client, model, prompt, system, max_tokens, temperature, json_mode):
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = dict(model=model, messages=messages,
                                  max_completion_tokens=max_tokens, temperature=temperature)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}   # prompt must mention JSON
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or "", resp.usage


def _anthropic(model, prompt, system, max_tokens, temperature, api_key, json_mode) -> str:
    api_key = api_key or _env("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError("anthropic is not configured: set ANTHROPIC_API_KEY")
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise LLMError("the 'anthropic' package is not installed (pip install anthropic)") from e
    with Anthropic(api_key=api_key, timeout=60.0, max_retries=1) as client:
        return _anthropic_request(client, model, prompt, system, max_tokens, temperature, json_mode)[0]


def _anthropic_request(client, model, prompt, system, max_tokens, temperature, json_mode):
    if json_mode:
        prompt += "\nReturn only one JSON object."
    kwargs: dict[str, Any] = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                                  messages=[{"role": "user", "content": prompt}])
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text"), resp.usage


_PROVIDERS: dict[str, ProviderFn] = {"fake": _fake, "openai": _openai, "anthropic": _anthropic}


def register_provider(name: str, fn: ProviderFn) -> None:
    """Add a provider (e.g. a local Ollama server) without editing this file's logic."""
    _PROVIDERS[name.lower()] = fn


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------

class LLMClient:
    """A client bound to one provider + model (one per agent if you like)."""

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 provider: str | None = None, agent: str | None = None,
                 timeout: float | None = None, max_retries: int | None = None):
        self.api_key = api_key
        self.agent = agent
        self.provider, self.model = _resolve(agent, provider, model)
        if self.provider not in _PROVIDERS:
            raise LLMError(f"unknown provider '{self.provider}'")
        if isinstance(timeout, bool) or (max_retries is not None and type(max_retries) is not int):
            raise LLMError("timeout must be numeric and max_retries must be an integer")
        try:
            self.timeout = float(timeout if timeout is not None else (_env("LLM_TIMEOUT_SECONDS") or 60))
            self.max_retries = int(max_retries if max_retries is not None else (_env("LLM_MAX_RETRIES") or 1))
        except (TypeError, ValueError) as exc:
            raise LLMError("LLM timeout/retry settings must be numeric") from exc
        if not math.isfinite(self.timeout) or not 0 < self.timeout <= 600 or not 0 <= self.max_retries <= 3:
            raise LLMError("LLM timeout must be 0..600 seconds and retries 0..3")
        self._sdk = None
        self._lock = RLock()
        self._closed = False
        self._usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "elapsed_seconds": 0.0}
        self._local = local()

    def __repr__(self) -> str:
        return f"LLMClient(agent={self.agent!r}, provider={self.provider!r}, model={self.model!r})"

    def _raw(self, prompt: str, system: str | None, max_tokens: int,
             temperature: float, json_mode: bool) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise LLMError("prompt must be a non-empty string")
        if self._closed:
            raise LLMError("LLM client is closed")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 32000:
            raise LLMError("max_tokens must be an integer between 1 and 32000")
        fn = _PROVIDERS.get(self.provider)
        if fn is None:
            raise LLMError(f"unknown provider '{self.provider}' (known: {sorted(_PROVIDERS)})")
        started = perf_counter()
        usage = None
        try:
            if (self.provider == "openai" and fn is _openai) or (self.provider == "anthropic" and fn is _anthropic):
                client = self._get_sdk()
                request = _openai_request if self.provider == "openai" else _anthropic_request
                out, usage = request(client, self.model, prompt, system, max_tokens, temperature, json_mode)
            else:
                out = fn(self.model, prompt, system, max_tokens, temperature, self.api_key, json_mode)
        except LLMError:
            raise
        except Exception as e:      # SDK / network / auth errors
            # Keep SDK messages (which may contain request data/credentials) out of UI logs.
            raise LLMError(f"{self.provider} call failed ({type(e).__name__})") from e
        finally:
            with self._lock:
                delta = {"calls": 1, "elapsed_seconds": perf_counter() - started,
                         "input_tokens": (getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)) or 0),
                         "output_tokens": (getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0)) or 0)}
                current = getattr(self._local, "usage", dict.fromkeys(delta, 0))
                for key, value in delta.items():
                    self._usage[key] += value
                    current[key] += value
                self._local.usage = current
        if not isinstance(out, str):
            raise LLMError(f"{self.provider} returned {type(out).__name__}, expected text")
        return out

    def _get_sdk(self):
        with self._lock:
            if self._sdk is None:
                env_key = "OPENAI_API_KEY" if self.provider == "openai" else "ANTHROPIC_API_KEY"
                key = self.api_key or _env(env_key)
                if not key:
                    raise LLMError(f"{self.provider} is not configured: set {env_key}")
                try:
                    if self.provider == "openai":
                        from openai import OpenAI as SDK
                    else:
                        from anthropic import Anthropic as SDK
                except ImportError as exc:
                    raise LLMError(f"The {self.provider} SDK is not installed") from exc
                self._sdk = SDK(api_key=key, timeout=self.timeout, max_retries=self.max_retries)
            return self._sdk

    def usage(self):
        """Counters for this thread; query deltas exclude concurrent callers."""
        return dict(getattr(self._local, "usage", dict.fromkeys(self._usage, 0)))

    def total_usage(self):
        with self._lock:
            return dict(self._usage)

    def close(self):
        with self._lock:
            self._closed = True
            if self._sdk is not None:
                self._sdk.close()
                self._sdk = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 1000,
                 temperature: float = 0.0) -> str:
        return self._raw(prompt, system, max_tokens, temperature, json_mode=False)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Backward-compatible alias for complete()."""
        return self.complete(prompt, **kwargs)

    def complete_json(self, prompt: str, system: str | None = None, max_tokens: int = 1000) -> dict:
        """Return a parsed JSON object. Retries ONCE on bad JSON, then raises LLMError."""
        last = ""
        for attempt in range(2):
            p = prompt if attempt == 0 else (
                prompt + "\n\nYour previous reply was not valid JSON. "
                         "Return ONLY one valid JSON object and nothing else.")
            text = self._raw(p, system, max_tokens, 0.0, json_mode=True)
            try:
                return _parse_json_object(text)
            except ValueError as e:
                last = str(e)
        raise LLMError(f"model did not return a valid JSON object after one retry: {last}")


# ---------------------------------------------------------------------------
# Module-level API (matches the contract) + per-agent clients
# ---------------------------------------------------------------------------

def for_agent(agent: str) -> LLMClient:
    """Client for one agent, e.g. for_agent('reasoning'), for_agent('critic')."""
    if not isinstance(agent, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", agent):
        raise LLMError("agent name must be letters/digits/underscore, e.g. 'reasoning'")
    return LLMClient(agent=agent)


def complete(prompt: str, system: str | None = None, max_tokens: int = 1000,
             temperature: float = 0.0) -> str:
    with LLMClient() as client:
        return client.complete(prompt, system, max_tokens, temperature)


def complete_json(prompt: str, system: str | None = None, max_tokens: int = 1000) -> dict:
    with LLMClient() as client:
        return client.complete_json(prompt, system, max_tokens)
