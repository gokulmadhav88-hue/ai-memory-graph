"""Neo4j graph wrapper."""

from __future__ import annotations


class Neo4jStore:
    """Simple placeholder wrapper around a Neo4j graph database."""

    def __init__(self, uri: str | None = None, username: str | None = None, password: str | None = None):
        self.uri = uri
        self.username = username
        self.password = password

    def connect(self):
        return True

    def write(self, records):
        return {"written": len(records)}

    def query(self, cypher: str):
        return {"cypher": cypher, "results": []}
