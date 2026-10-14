"""Transactional Neo4j adapter for a small corpus.

State and its browsable graph projection commit together. State is loaded in
memory; revision checks reject stale writers. Every record is scoped by graph_id.
"""
from __future__ import annotations

import json
import os

from src.ingestion.graph_agent import MemoryStore
from .errors import StoreError


class Neo4jStore(MemoryStore):
    def __init__(self, uri=None, username=None, password=None, *, database=None,
                 graph_id=None, driver=None):
        super().__init__()
        self.uri = uri or os.getenv('NEO4J_URI', 'neo4j://localhost:7687')
        self.username = username or os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD')
        self.database = database or os.getenv('NEO4J_DATABASE', 'neo4j')
        self.graph_id = graph_id or os.getenv('NEO4J_GRAPH_ID', 'memory_graph')
        self._driver = driver
        self._connected = False
        self._revision = 0
        self._committed_state = self.snapshot()

    def connect(self):
        if self._connected:
            return True
        try:
            if self._driver is None:
                if not self.password:
                    raise StoreError('Set NEO4J_PASSWORD in .env before connecting to Neo4j')
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password),
                                                   connection_timeout=10)
            self._driver.verify_connectivity()
            with self._driver.session(database=self.database) as session:
                for label in ('AMGState', 'AMGEntity', 'AMGMemory', 'AMGNote', 'AMGAudit'):
                    session.run(f'CREATE CONSTRAINT {label.lower()}_key IF NOT EXISTS '
                                f'FOR (n:{label}) REQUIRE (n.graph_id, n.id) IS UNIQUE').consume()
            self._connected = True
            self.refresh()
        except StoreError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise StoreError(f'Neo4j connection failed ({type(exc).__name__})') from exc
        return True

    def refresh(self):
        if not self._connected:
            self.connect()
            return
        try:
            with self.lock, self._driver.session(database=self.database) as session:
                row = session.execute_read(lambda tx: tx.run(
                    'MATCH (s:AMGState {graph_id:$graph_id, id:"state"}) '
                    'RETURN s.payload AS payload, s.revision AS revision',
                    graph_id=self.graph_id).single())
                if row:
                    data = json.loads(row['payload'])
                    if data.pop('version') != 1:
                        raise StoreError('Unsupported Neo4j memory schema version')
                    data.setdefault('documents', {})
                    self.restore(data)
                    self._revision = row['revision']
                else:
                    self.restore(MemoryStore().snapshot())
                    self._revision = 0
                self._committed_state = self.snapshot()
        except StoreError:
            raise
        except Exception as exc:
            raise StoreError(f'Neo4j read failed ({type(exc).__name__})') from exc

    def commit(self):
        if not self._connected:
            raise StoreError('Call connect() before mutating the Neo4j store')
        with self.lock:
            state = self.snapshot()
            if state == self._committed_state:
                return
            payload = json.dumps({'version': 1, **state}, ensure_ascii=False)
            changes = self._projection_changes(self._committed_state, state)
            try:
                with self._driver.session(database=self.database) as session:
                    session.execute_write(self._persist, state, payload, self._revision, changes)
                self._revision += 1
                self._committed_state = state
            except StoreError:
                raise
            except Exception as exc:
                raise StoreError(f'Neo4j commit failed ({type(exc).__name__})') from exc

    @staticmethod
    def _projection_changes(before, after):
        """The v1 snapshot stays intact; only changed projection records are sent."""
        changes = {}
        for key in ('nodes', 'memories', 'notes', 'events', 'edges'):
            old, new = before[key], after[key]
            if key == 'events':
                old = {str(i): e for i, e in enumerate(old)}
                new = {str(i): e for i, e in enumerate(new)}
            if key == 'edges':
                old = {eid: dict(e, status=before['memories'][e['memory_id']]['status']) for eid, e in old.items()}
                new = {eid: dict(e, status=after['memories'][e['memory_id']]['status']) for eid, e in new.items()}
            changes[key] = {'rows': {rid: r for rid, r in new.items() if r != old.get(rid)},
                            'deleted': sorted(set(old) - set(new))}
        return changes

    def _persist(self, tx, state, payload, revision, changes):
        params = {'graph_id': self.graph_id}
        # Lock before inspecting revision to prevent lost updates.
        row = tx.run('MERGE (s:AMGState {graph_id:$graph_id, id:"state"}) '
                     'ON CREATE SET s.revision=0 '
                     'SET s.lock_version=coalesce(s.lock_version,0)+1 '
                     'RETURN s.revision AS revision', **params).single()
        if row['revision'] != revision:
            raise StoreError('Neo4j changed in another session. Refresh and retry the operation.')
        tx.run('MATCH (s:AMGState {graph_id:$graph_id, id:"state"}) '
               'SET s.payload=$payload, s.revision=$revision',
               payload=payload, revision=revision + 1, **params).consume()
        for label, mapping_key, key, visible in (
            ('AMGEntity', 'nodes', 'node_id', ('name', 'entity_type', 'aliases')),
            ('AMGMemory', 'memories', 'memory_id', ('content', 'status', 'event_time')),
            ('AMGNote', 'notes', 'note_id', ('doc_id', 'content', 'source_filename')),
            ('AMGAudit', 'events', None, ('action', 'reason')),
        ):
            rows = [{'id': record[key] if key else rid,
                     'props': {k: record.get(k) for k in visible} | {'payload': json.dumps(record)}}
                    for rid, record in changes[mapping_key]['rows'].items()]
            if rows:
                tx.run(f'UNWIND $rows AS row MERGE (n:{label} {{graph_id:$graph_id,id:row.id}}) '
                       'SET n += row.props', rows=rows, **params).consume()
            if changes[mapping_key]['deleted']:
                tx.run(f'MATCH (n:{label} {{graph_id:$graph_id}}) WHERE n.id IN $ids '
                       'DETACH DELETE n', ids=changes[mapping_key]['deleted'], **params).consume()
        # Fixed relationship type; extracted relation names are parameter values.
        if changes['edges']['deleted']:
            tx.run('MATCH (:AMGEntity {graph_id:$graph_id})-[r:AMG_FACT]->() '
                   'WHERE r.edge_id IN $ids DELETE r', ids=changes['edges']['deleted'], **params).consume()
        if changes['edges']['rows']:
            tx.run('UNWIND $rows AS row '
                   'MATCH (a:AMGEntity {graph_id:$graph_id,id:row.source_node_id}) '
                   'MATCH (b:AMGEntity {graph_id:$graph_id,id:row.target_node_id}) '
                   'MERGE (a)-[r:AMG_FACT {edge_id:row.edge_id}]->(b) '
                   'SET r.relation_type=row.relation_type, r.memory_id=row.memory_id, '
                   'r.source_note_ids=row.source_note_ids, r.status=row.status',
                   rows=list(changes['edges']['rows'].values()), **params).consume()
        support_rows = [n for nid, n in changes['notes']['rows'].items()
                        if self._committed_state['notes'].get(nid, {}).get('memory_id') != n['memory_id']]
        if support_rows:
            # A consolidation may reassign a note to a different canonical memory.
            tx.run('UNWIND $rows AS row MATCH (n:AMGNote {graph_id:$graph_id,id:row.note_id}) '
                   'OPTIONAL MATCH (n)-[r:AMG_SUPPORTS]->() DELETE r', rows=support_rows, **params).consume()
            tx.run('UNWIND $rows AS row '
                   'MATCH (n:AMGNote {graph_id:$graph_id,id:row.note_id}) '
                   'MATCH (m:AMGMemory {graph_id:$graph_id,id:row.memory_id}) '
                   'MERGE (n)-[:AMG_SUPPORTS]->(m)', rows=support_rows, **params).consume()

    def query(self, cypher, parameters=None):
        """Diagnostic query API, never called with model-generated Cypher."""
        self.connect()
        try:
            with self._driver.session(database=self.database) as session:
                return session.execute_read(lambda tx: tx.run(cypher, parameters or {}).data())
        except Exception as exc:
            raise StoreError(f'Neo4j query failed ({type(exc).__name__})') from exc

    def close(self):
        if self._driver is not None:
            self._driver.close()
        self._driver = None
        self._connected = False
