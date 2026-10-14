"""Adapter protocol tests; live round-trip is opt-in with RUN_NEO4J_TESTS=1."""
from contextlib import contextmanager
from copy import deepcopy
import os
import uuid

import pytest

from src.ingestion.graph_agent import GraphAgent
from src.stores.errors import StoreError
from src.stores.neo4j_store import Neo4jStore
from test_graph_agent import note


class Result:
    def __init__(self, row=None):
        self.row = row

    def single(self):
        return self.row

    def consume(self):
        return None


class Driver:
    """Emulates commit/rollback and revision storage, not Cypher execution."""
    def __init__(self):
        self.state = {}
        self.statements = []
        self.fail = False

    def verify_connectivity(self):
        pass

    @contextmanager
    def session(self, **kwargs):
        yield self

    def execute_read(self, callback):
        return callback(self)

    def execute_write(self, callback, *args):
        before = deepcopy(self.state)
        try:
            return callback(self, *args)
        except Exception:
            self.state = before
            raise

    def run(self, query, **params):
        self.statements.append((query, params))
        if 'CREATE CONSTRAINT' in query:
            return Result()
        key = params['graph_id']
        if 'RETURN s.payload' in query:
            return Result(self.state.get(key))
        if 'RETURN s.revision' in query:
            return Result({'revision': self.state.get(key, {}).get('revision', 0)})
        if 'SET s.payload' in query:
            self.state[key] = {'payload': params['payload'], 'revision': params['revision']}
        elif self.fail:
            raise RuntimeError('projection write failed')
        return Result()

    def close(self):
        pass


def test_neo4j_round_trip_preserves_provenance_and_outbox():
    driver = Driver()
    store = Neo4jStore(driver=driver)
    store.connect()
    GraphAgent(store).ingest_note(note())
    reopened = Neo4jStore(driver=driver)
    reopened.connect()
    assert reopened.snapshot() == store.snapshot()
    assert reopened.query_subgraph(['Atlas'])['edges']
    assert next(iter(reopened.notes.values()))['vector_pending']
    assert any('AMG_FACT' in q for q, p in driver.statements)


def test_stale_neo4j_writer_rejected_without_data_loss():
    driver = Driver()
    a, b = Neo4jStore(driver=driver), Neo4jStore(driver=driver)
    a.connect()
    b.connect()
    GraphAgent(a).ingest_note(note())
    with pytest.raises(StoreError, match='another session'):
        GraphAgent(b).ingest_note(note('second'))
    assert b.notes == {}
    b.refresh()
    assert b.snapshot() == a.snapshot()


def test_failed_projection_rolls_back_state_and_graph_agent():
    driver = Driver()
    store = Neo4jStore(driver=driver)
    store.connect()
    before = store.snapshot()
    driver.fail = True
    with pytest.raises(StoreError):
        GraphAgent(store).ingest_note(note())
    assert store.snapshot() == before and not driver.state


def test_neo4j_requires_credentials(monkeypatch):
    monkeypatch.delenv('NEO4J_PASSWORD', raising=False)
    with pytest.raises(StoreError, match='NEO4J_PASSWORD'):
        Neo4jStore().connect()


@pytest.mark.skipif(os.getenv('RUN_NEO4J_TESTS') != '1',
                    reason='Set RUN_NEO4J_TESTS=1 with configured Neo4j for live validation')
def test_live_neo4j_round_trip():
    from dotenv import load_dotenv
    load_dotenv()
    graph_id = 'integration_test_' + uuid.uuid4().hex
    store = Neo4jStore(graph_id=graph_id)
    try:
        store.connect()
        GraphAgent(store).ingest_note(note())
        reopened = Neo4jStore(graph_id=graph_id)
        try:
            reopened.connect()
            assert reopened.snapshot() == store.snapshot()
            assert reopened.query_subgraph(['Atlas'])['edges']
        finally:
            reopened.close()
    finally:
        # Deletes only records created by this test, using its unique namespace.
        if store._connected:
            with store._driver.session(database=store.database) as session:
                session.execute_write(lambda tx: tx.run(
                    'MATCH (n {graph_id:$graph_id}) DETACH DELETE n', graph_id=graph_id).consume())
        store.close()
