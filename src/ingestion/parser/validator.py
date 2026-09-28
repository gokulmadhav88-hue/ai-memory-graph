"""File checks (rules 1, 3, 5, 6). Fails early with a clear message."""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import config


class ParserError(Exception):
    """Base class for every parser error."""


class UnsupportedFileType(ParserError):
    pass


class FileTooLarge(ParserError):
    pass


class EncryptedFile(ParserError):
    pass


class InvalidFile(ParserError):
    pass


def validate_file(file_path: str | Path) -> Path:
    """Check the file is real, supported, small enough, and matches its extension."""
    path = Path(file_path)

    if not path.is_file():
        raise InvalidFile(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(
            f"{path.name}: '{ext}' is not supported. "
            f"Supported types: {sorted(config.SUPPORTED_EXTENSIONS)}"
        )

    if path.stat().st_size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise FileTooLarge(f"{path.name} is larger than {config.MAX_FILE_SIZE_MB} MB")

    with open(path, "rb") as f:
        head = f.read(4096)

    if ext == ".pdf":
        if not head.startswith(b"%PDF"):
            raise InvalidFile(f"{path.name} has a .pdf extension but is not a real PDF")
        if b"/Encrypt" in path.read_bytes():   # simple check, not perfect
            raise EncryptedFile(f"{path.name} is encrypted or password-protected")
    elif ext in (".docx", ".xlsx"):
        if head.startswith(b"\xd0\xcf\x11\xe0"):
            raise EncryptedFile(f"{path.name} is encrypted or in an old Office format")
        if not head.startswith(b"PK"):
            raise InvalidFile(f"{path.name} has a {ext} extension but is not a real {ext} file")
    elif ext in (".txt", ".md", ".csv"):
        if b"\x00" in head:
            raise InvalidFile(f"{path.name} looks like a binary file, not text")

    return path


def file_hash(file_path: str | Path) -> str:
    """SHA-256 of the whole file, used as content_hash on every chunk (rule 6)."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()