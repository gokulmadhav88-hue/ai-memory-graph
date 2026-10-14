"""Errors shared by storage adapters."""


class StoreError(Exception):
    """A storage operation failed (empty search results are not failures)."""
