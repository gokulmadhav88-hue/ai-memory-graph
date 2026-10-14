"""Validated runtime controls. Configuration never silently disables team access checks."""
from dataclasses import dataclass
import math
import os


class ConfigurationError(ValueError):
    pass


def boolean(name, default=False):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    raw = raw.strip().lower()
    if raw not in ('1', 'true', 'yes', 'on', '0', 'false', 'no', 'off'):
        raise ConfigurationError(f'{name} must be true or false')
    return raw in ('1', 'true', 'yes', 'on')


def number(name, default, minimum, maximum, *, integer=True):
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw) if integer else float(raw)
    except ValueError as exc:
        raise ConfigurationError(f'{name} must be numeric') from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ConfigurationError(f'{name} must be between {minimum} and {maximum}')
    return value


@dataclass(frozen=True)
class Settings:
    app_env: str = 'local'
    auth_required: bool = False
    allowed_emails: tuple = ()
    top_k: int = 5
    graph_depth: int = 2
    min_score: float = 0.25
    max_graph_facts: int = 20
    max_retries: int = 2
    max_upload_files: int = 10
    max_history_turns: int = 20
    enable_ocr: bool = False

    @classmethod
    def from_env(cls):
        app_env = os.getenv('APP_ENV', 'local').strip().lower()
        if app_env not in ('local', 'team'):
            raise ConfigurationError('APP_ENV must be local or team; public user isolation is not implemented')
        auth = boolean('APP_AUTH_REQUIRED', app_env == 'team')
        emails = tuple(dict.fromkeys(x.strip().casefold() for x in
                                    os.getenv('APP_ALLOWED_EMAILS', '').split(',') if x.strip()))
        if app_env == 'team' and not auth:
            raise ConfigurationError('Team mode requires APP_AUTH_REQUIRED=1')
        if auth and not emails:
            raise ConfigurationError('Set APP_ALLOWED_EMAILS to the team members permitted to access this workspace')
        return cls(app_env, auth, emails,
                   number('RETRIEVAL_TOP_K', 5, 1, 100),
                   number('RETRIEVAL_GRAPH_DEPTH', 2, 1, 5),
                   number('RETRIEVAL_MIN_SCORE', 0.25, 0, 1, integer=False),
                   number('RETRIEVAL_MAX_GRAPH_FACTS', 20, 1, 100),
                   number('QUERY_MAX_RETRIES', 2, 0, 2),
                   number('APP_MAX_UPLOAD_FILES', 10, 1, 50),
                   number('APP_MAX_HISTORY_TURNS', 20, 2, 100),
                   boolean('PARSER_ENABLE_OCR'))
