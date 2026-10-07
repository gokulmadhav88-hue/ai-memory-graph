from copy import deepcopy
import pytest
from src.ingestion.extraction_agent import ExtractionAgent
from src.ingestion.graph_agent import GraphAgent, MemoryStore, MemoryIntegrationError
from test_extraction_agent import FakeLLM, output

def note(doc='doc', time='2026-09-01T00:00:00Z', person='Maya'):
    text = f'Atlas is managed by {person}.'
    raw = output(text, time)
    raw['entities'][1]['name'] = person
    raw['relations'][0]['object'] = person
    return ExtractionAgent(FakeLLM([raw])).extract(text, doc)

class DecisionLLM:
    def __init__(self, store, action):
        self.store, self.action = store, action
    def complete_json(self, prompt, **kwargs):
        if 'candidates' in __import__('json').loads(prompt):
            return {'node_id': None}
        return {'action': self.action, 'target_ids': [next(iter(self.store.memories))], 'reason': 'New evidence'}

def test_idempotent_ingest_and_merge_provenance():
    agent = GraphAgent()
    first = note()
    agent.ingest_note(first)
    assert agent.ingest_note(first)['skip_reason'] == 'duplicate_note'
    agent.ingest_note(note('another'))
    assert len(agent.graph_store.memories) == 1
    assert len(next(iter(agent.graph_store.memories.values()))['source_note_ids']) == 2

def test_update_preserves_history():
    store = MemoryStore()
    agent = GraphAgent(store)
    old = agent.ingest_note(note())['memory_id']
    agent.llm = DecisionLLM(store, 'update')
    new = agent.ingest_note(note('new', '2026-10-01T00:00:00Z', 'Arun'))['memory_id']
    assert store.memories[old]['status'] == 'superseded'
    assert store.memories[new]['supersedes'] == [old]
    assert store.memories[old]['revisions'] and len(store.notes) == 2

def test_older_upload_does_not_overwrite():
    store = MemoryStore()
    agent = GraphAgent(store)
    old = agent.ingest_note(note())['memory_id']
    agent.llm = DecisionLLM(store, 'update')
    result = agent.ingest_note(note('old', '2026-08-01T00:00:00Z', 'Arun'))
    assert result['decision']['action'] == 'add'
    assert store.memories[old]['status'] == 'active'

def test_retire_cannot_remove_conflicting_facts():
    store = MemoryStore()
    agent = GraphAgent(store)
    old = agent.ingest_note(note())['memory_id']
    agent.llm = DecisionLLM(store, 'retire')
    result = agent.ingest_note(note('new', person='Arun'))
    assert result['decision']['action'] == 'add' and store.memories[old]['status'] == 'active'

def test_persistent_store(tmp_path):
    path = tmp_path / 'memory.json'
    agent = GraphAgent(MemoryStore(path))
    agent.ingest_note(note())
    reopened = MemoryStore(path)
    assert len(reopened.notes) == len(reopened.active_memories()) == 1

def test_pending_vector_retry():
    class Vector:
        def add(self, *args, **kwargs):
            raise RuntimeError('offline')
    agent = GraphAgent(vector_store=Vector())
    agent.ingest_note(note())
    assert next(iter(agent.graph_store.notes.values()))['vector_pending']
    assert agent.flush_vector_outbox()[0]['error'] == 'offline'

def test_invalid_decision_rolls_back():
    store = MemoryStore()
    agent = GraphAgent(store)
    agent.ingest_note(note())
    before = store.snapshot()
    class Bad:
        def complete_json(self, prompt, **kwargs):
            if 'candidates' in __import__('json').loads(prompt):
                return {'node_id': None}
            return {'action': 'update', 'target_ids': ['unknown'], 'reason': 'bad'}
    agent.llm = Bad()
    with pytest.raises(MemoryIntegrationError):
        agent.ingest_note(note('new', person='Arun'))
    assert store.snapshot() == before

def test_add_retain_and_retire_equivalent_memories():
    for action in ('add', 'retain', 'retire'):
        store = MemoryStore()
        agent = GraphAgent(store)
        old = agent.ingest_note(note())['memory_id']
        incoming = note('new')
        incoming.content += ' Confirmed by a second report.'
        agent.llm = DecisionLLM(store, action)
        result = agent.ingest_note(incoming)
        assert result['decision']['action'] == action
        assert store.memories[old]['status'] == ('retired' if action == 'retire' else 'active')
        assert len(store.notes) == 2

def test_alias_resolution():
    store = MemoryStore()
    agent = GraphAgent(store)
    agent.ingest_note(note())
    maya_id = next(n['node_id'] for n in store.nodes.values() if n['name'] == 'Maya')
    class Alias:
        def complete_json(self, prompt, **kwargs):
            data = __import__('json').loads(prompt)
            if 'candidates' in data:
                return {'node_id': maya_id}
            return {'action': 'merge', 'target_ids': [next(iter(store.memories))], 'reason': 'Same person'}
    agent.llm = Alias()
    agent.ingest_note(note('new', person='Maya Smith'))
    assert len(store.nodes) == 2
    assert 'Maya Smith' in store.nodes[maya_id]['aliases']

def test_multivalued_relation_cannot_supersede():
    store = MemoryStore()
    agent = GraphAgent(store, exclusive_relations=[])
    old = agent.ingest_note(note())['memory_id']
    agent.llm = DecisionLLM(store, 'update')
    result = agent.ingest_note(note('new', '2026-10-01T00:00:00Z', 'Arun'))
    assert result['decision']['action'] == 'add'
    assert store.memories[old]['status'] == 'active'
