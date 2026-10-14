"""Persistence failures, document boundaries, and incremental projection regressions."""
from copy import deepcopy
import json
from threading import Event, Thread

import pytest

from src.demo import DemoLLM
from src.ingestion.extraction_agent import ExtractionAgent
from src.ingestion.graph_agent import GraphAgent, MemoryStore
from src.ingestion.parser.schema import Chunk
from src.ingestion.parser.validator import file_hash
from src.ingestion.pipeline import IngestionPipeline
from src.stores.neo4j_store import Neo4jStore
from src.stores.errors import StoreError
from src.stores.vector_store import VectorStore
from test_graph_agent import DecisionLLM, note
from test_neo4j_adapter import Driver


class CountingStore(MemoryStore):
    def __init__(self, path=None):
        super().__init__(path)
        self.commits = 0
        self.fail_commit = False

    def commit(self):
        if self.fail_commit:
            raise RuntimeError('database unavailable')
        self.commits += 1
        super().commit()


class BatchedVector:
    def __init__(self):
        self.records = {}
        self.batches = []
        self.deletions = []
        self.fail_delete = False

    def add_many(self, records):
        self.batches.append(deepcopy(records))
        for record in records:
            self.records[record['note_id']] = deepcopy(record)

    def delete_notes(self, ids):
        if self.fail_delete:
            raise RuntimeError('vector deletion unavailable')
        self.deletions.append(list(ids))
        for nid in ids:
            self.records.pop(nid, None)


def make_pipeline(store=None, vector=None):
    store = store or CountingStore()
    vector = vector or BatchedVector()
    return IngestionPipeline(ExtractionAgent(DemoLLM('extraction')), GraphAgent(store, vector))


def make_file(tmp_path):
    path = tmp_path / 'atlas.txt'
    path.write_text('Atlas is managed by Maya.', encoding='utf-8')
    return path


def test_document_batches_graph_commits_and_vector_embeddings(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    chunks = [Chunk(text=path.read_text(), doc_id='atlas', chunk_index=i,
                    content_hash=file_hash(path), source_filename=path.name, parser_version='test')
              for i in range(12)]
    monkeypatch.setattr('src.ingestion.pipeline.parse_document', lambda _: chunks)
    pipeline = make_pipeline()
    report = pipeline.ingest_document(path)
    assert report['status'] == 'ingested' and report['vector_pending'] == 0
    assert pipeline.graph_agent.graph_store.commits == 2  # document + outbox checkpoint
    assert len(pipeline.graph_agent.vector_store.batches) == 1
    assert len(pipeline.graph_agent.vector_store.batches[0]) == 12
    assert report['edges_created'] == 1  # merging support does not create more edges


def test_new_document_commit_failure_restores_memory(tmp_path):
    path = make_file(tmp_path)
    store, vector = CountingStore(), BatchedVector()
    store.fail_commit = True
    before = store.snapshot()
    report = make_pipeline(store, vector).ingest_document(path)
    assert report['status'] == 'failed'
    assert store.snapshot() == before and vector.records == {}


def test_empty_replacement_preserves_committed_document(tmp_path, monkeypatch):
    path = make_file(tmp_path)
    pipeline = make_pipeline()
    assert pipeline.ingest_document(path)['status'] == 'ingested'
    store = pipeline.graph_agent.graph_store
    before = store.snapshot()
    path.write_text('No readable content.', encoding='utf-8')
    monkeypatch.setattr('src.ingestion.pipeline.parse_document', lambda _: [])
    report = pipeline.ingest_document(path, replace=True)
    assert report['status'] == 'failed' and 'No readable text' in report['errors'][0]
    assert store.snapshot() == before


def test_uninformative_replacement_preserves_committed_document(tmp_path):
    path = make_file(tmp_path)
    pipeline = make_pipeline()
    pipeline.ingest_document(path)
    before = pipeline.graph_agent.graph_store.snapshot()
    path.write_text('tiny noise', encoding='utf-8')
    report = pipeline.ingest_document(path, replace=True)
    assert report['status'] == 'failed'
    assert pipeline.graph_agent.graph_store.snapshot() == before


def test_failed_vector_cleanup_survives_reopen_and_retries(tmp_path):
    path = make_file(tmp_path)
    store = CountingStore(tmp_path / 'memory.json')
    vector = BatchedVector()
    pipeline = make_pipeline(store, vector)
    pipeline.ingest_document(path)
    old_ids = set(store.notes)
    vector.fail_delete = True
    path.write_text('Atlas is managed by Arun.', encoding='utf-8')
    report = pipeline.ingest_document(path, replace=True)
    assert report['status'] == 'replaced' and report['vector_cleanup_pending'] == 1
    assert report['vector_pending'] == 1
    reopened = CountingStore(store.path)
    assert set(reopened.documents['atlas']['vector_delete_note_ids']) == old_ids
    vector.fail_delete = False
    retry = make_pipeline(reopened, vector).ingest_document(path)
    assert retry['status'] == 'skipped_duplicate'
    assert retry['vector_cleanup_pending'] == retry['vector_pending'] == 0
    assert set(vector.records) == set(reopened.notes)


def test_outbox_ack_failure_can_safely_replay_vector_upserts():
    store, vector = CountingStore(), BatchedVector()
    agent = GraphAgent(store, vector)
    agent.ingest_note(note(), defer_commit=True)
    store.commit()
    store.fail_commit = True
    errors = agent.flush_vector_outbox()
    assert errors[-1]['operation'] == 'checkpoint_vector_outbox'
    assert next(iter(store.notes.values()))['vector_pending']
    assert len(vector.records) == 1
    store.fail_commit = False
    assert agent.flush_vector_outbox() == []
    assert not next(iter(store.notes.values()))['vector_pending'] and len(vector.records) == 1


def test_extraction_does_not_hold_graph_lock(tmp_path):
    path = make_file(tmp_path)
    store = CountingStore()
    pipeline = make_pipeline(store)
    original = pipeline.extractor.extract_note

    def extract(chunk):
        acquired = Event()

        def reader():
            with store.lock:
                acquired.set()

        thread = Thread(target=reader, daemon=True)
        thread.start()
        assert acquired.wait(1), 'Reader blocked for the duration of extraction'
        thread.join(1)
        return original(chunk)

    pipeline.extractor.extract_note = extract
    assert pipeline.ingest_document(path)['status'] == 'ingested'


def test_retired_equivalent_source_survives_replacement_source_deletion():
    store = MemoryStore()
    agent = GraphAgent(store)
    first = note('first')
    agent.ingest_note(first)
    agent.llm = DecisionLLM(store, 'retire')
    second = note('second')
    second.content += ' Confirmed independently.'
    agent.ingest_note(second)
    store.delete_document('second')
    assert store.is_note_active(first.note_id)
    active_edges = store.query_subgraph(['Atlas'])['edges']
    assert len(active_edges) == 1 and active_edges[0]['source_note_ids'] == [first.note_id]
    assert store.notes[first.note_id]['memory_id'] in store.memories


def test_superseded_vectors_are_removed_without_losing_source_history():
    store, vector = CountingStore(), BatchedVector()
    agent = GraphAgent(store, vector)
    first = note('first')
    agent.ingest_note(first)
    agent.llm = DecisionLLM(store, 'update')
    second = note('second', '2026-10-01T00:00:00Z', 'Arun')
    agent.ingest_note(second)
    assert set(vector.records) == {second.note_id}
    assert set(store.notes) == {first.note_id, second.note_id}
    assert not store.is_note_active(first.note_id)


def test_superseded_vector_deletion_failure_indexes_current_and_retries():
    store, vector = CountingStore(), BatchedVector()
    agent = GraphAgent(store, vector)
    first = note('first')
    agent.ingest_note(first)
    agent.llm = DecisionLLM(store, 'update')
    vector.fail_delete = True
    second = note('second', '2026-10-01T00:00:00Z', 'Arun')
    agent.ingest_note(second)
    assert set(vector.records) == {first.note_id, second.note_id}
    assert store.notes[first.note_id]['vector_delete_pending']
    assert not store.notes[second.note_id]['vector_pending']
    assert agent.flush_vector_outbox()[0]['operation'] == 'delete_superseded_vectors'
    vector.fail_delete = False
    assert agent.flush_vector_outbox() == []
    assert set(vector.records) == {second.note_id}
    assert 'vector_delete_pending' not in store.notes[first.note_id]


def test_graph_write_batches_commits_and_rolls_back_failed_batch():
    store, vector = CountingStore(), BatchedVector()
    agent = GraphAgent(store, vector)
    result = agent.write([note('first'), note('second')])
    assert result['written'] == 2 and store.commits == 2 and len(vector.batches) == 1
    before = store.snapshot()
    store.fail_commit = True
    with pytest.raises(RuntimeError, match='database unavailable'):
        agent.write([note('third', person='Arun')])
    assert store.snapshot() == before


def test_graph_prompts_do_not_expand_with_source_and_revision_history():
    store = MemoryStore()
    agent = GraphAgent(store)
    agent.write([note('first'), note('second'), note('third')])

    class InspectLLM:
        def complete_json(self, prompt, **kwargs):
            data = json.loads(prompt)
            if 'candidates' in data:
                assert all('source_note_ids' not in candidate for candidate in data['candidates'])
                return {'node_id': None}
            assert all('revisions' not in m and 'source_note_ids' not in m for m in data['memories'])
            return {'action': 'add', 'target_ids': [], 'reason': 'Keep distinct source'}

    agent.llm = InspectLLM()
    agent.ingest_note(note('fourth', person='Arun'))
    assert len(store.notes) == 4


def test_neo4j_outbox_checkpoint_only_projects_changed_notes():
    driver = Driver()
    store = Neo4jStore(driver=driver)
    store.connect()
    agent = GraphAgent(store, BatchedVector())
    agent.ingest_note(note(), defer_commit=True)
    store.commit()
    driver.statements.clear()
    agent.flush_vector_outbox()
    projection = [(q, p) for q, p in driver.statements if 'UNWIND' in q]
    assert len(projection) == 1 and 'AMGNote' in projection[0][0]
    assert len(projection[0][1]['rows']) == 1
    driver.statements.clear()
    store.commit()
    assert driver.statements == []  # no-op commit avoids revision and projection churn


def test_neo4j_delta_updates_superseded_edge_status():
    driver = Driver()
    store = Neo4jStore(driver=driver)
    store.connect()
    agent = GraphAgent(store)
    agent.ingest_note(note())
    driver.statements.clear()
    agent.llm = DecisionLLM(store, 'update')
    agent.ingest_note(note('second', '2026-10-01T00:00:00Z', 'Arun'))
    edge_rows = [r for q, p in driver.statements if 'SET r.relation_type' in q for r in p['rows']]
    assert {r['status'] for r in edge_rows} == {'active', 'superseded'}


def test_chroma_batch_upserts_and_exact_cleanup_preserve_new_version(tmp_path):
    pytest.importorskip('chromadb')
    from test_vector_store import FakeEmbedder
    vector = VectorStore('storage_hardening', persist_path=str(tmp_path / 'chroma'),
                         embedding_function=FakeEmbedder())
    vector.add_many([{'note_id': 'old', 'text': 'Atlas is managed by Maya.', 'metadata': {'doc_id': 'atlas'}},
                     {'note_id': 'new', 'text': 'Atlas is managed by Arun.', 'metadata': {'doc_id': 'atlas'}},
                     {'note_id': 'other', 'text': 'Nova has another manager.', 'metadata': {'doc_id': 'nova'}}],
                    batch_size=2)
    vector.delete_notes(['old'])
    assert {r['note_id'] for r in vector.query('manager', top_k=10)} == {'new', 'other'}


def test_chroma_rejects_bad_batch_before_writing(tmp_path):
    pytest.importorskip('chromadb')
    from test_vector_store import FakeEmbedder
    vector = VectorStore('storage_validation', persist_path=str(tmp_path / 'chroma'),
                         embedding_function=FakeEmbedder())
    with pytest.raises(StoreError, match='doc_id'):
        vector.add_many([{'note_id': 'valid', 'text': 'A valid note.', 'metadata': {'doc_id': 'atlas'}},
                         {'note_id': 'bad', 'text': 'A missing document identity.', 'metadata': {}}])
    assert vector.query('valid') == []
