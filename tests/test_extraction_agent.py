from copy import deepcopy
import pytest
from src.ingestion.extraction_agent import ExtractionAgent, ExtractionError

class FakeLLM:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
    def complete_json(self, *args, **kwargs):
        return deepcopy(next(self.outputs))

def output(text='Atlas is managed by Maya.', time=None):
    return {'is_informative': True, 'entities': [{'name': 'Atlas', 'type': 'Product'}, {'name': 'Maya', 'type': 'Person'}],
        'relations': [{'subject': 'Atlas', 'relation': 'managed_by', 'object': 'Maya', 'evidence_quote': text, 'event_time': time}],
        'keywords': ['Atlas'], 'tags': ['project'], 'context': text, 'title': 'Atlas leadership'}

def test_source_and_stable_distinct_ids():
    text = 'Atlas is managed by Maya.'
    agent = ExtractionAgent(FakeLLM([output(), output(), output()]))
    a = agent.extract(text, 'doc')
    b = agent.extract(text, 'doc')
    c = agent.extract_note({'text': text, 'doc_id': 'doc', 'chunk_index': 1})
    assert a.note_id == b.note_id != c.note_id
    assert a.content == text and a.source_document_id == 'doc'

def test_invalid_quote_repaired():
    bad = output('invented quote')
    agent = ExtractionAgent(FakeLLM([bad, output()]))
    assert agent.extract('Atlas is managed by Maya.', 'doc').is_informative

def test_invalid_output_fails_after_repair():
    with pytest.raises(ExtractionError):
        ExtractionAgent(FakeLLM([{}, {}])).extract('Useful text', 'doc')

def test_noise():
    raw = {'is_informative': False, 'entities': [], 'relations': [], 'keywords': [], 'tags': [], 'context': ''}
    assert not ExtractionAgent(FakeLLM([raw])).extract('Copyright notice', 'doc').is_informative

def test_conversation_metadata():
    text = 'user: Atlas is managed by Maya.'
    agent = ExtractionAgent(FakeLLM([output(text)]))
    note = agent.extract_conversation([{'role': 'user', 'content': 'Atlas is managed by Maya.', 'timestamp': '2026-10-01T00:00:00Z'}], 'chat')[0]
    assert note.source_type == 'conversation' and note.source_time.endswith('Z')

def test_unconfigured_client_explicit_error():
    with pytest.raises(ExtractionError, match='not configured'):
        ExtractionAgent().extract('Useful text', 'doc')
