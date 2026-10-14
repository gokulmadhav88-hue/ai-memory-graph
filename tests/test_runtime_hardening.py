from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from src import llm_client
from src.llm_client import LLMClient, LLMError
from src.settings import Settings, ConfigurationError
from app.access import authorized_identity
from src.runtime import create_runtime


def test_sdk_reused_with_bounded_options_and_closed(monkeypatch):
    import openai
    made = []
    class SDK:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.closed = False
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
            made.append(self)
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
                                   usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))
        def close(self):
            self.closed = True
    monkeypatch.setattr(openai, 'OpenAI', SDK)
    monkeypatch.delenv('LLM_FAKE', raising=False)
    client = LLMClient(provider='openai', model='test-model', api_key='test-only', timeout=12, max_retries=1)
    assert client.complete_json('JSON please') == client.complete_json('JSON again') == {'ok': True}
    assert len(made) == 1 and made[0].options['timeout'] == 12
    assert client.usage()['calls'] == 2 and client.usage()['input_tokens'] == 14
    client.close()
    assert made[0].closed
    with pytest.raises(LLMError, match='closed'):
        client.complete('test')


def test_thread_local_llm_usage_does_not_include_other_requests(monkeypatch):
    monkeypatch.delenv('LLM_FAKE', raising=False)
    monkeypatch.setitem(llm_client._PROVIDERS, 'counter_test', lambda *args: '{"ok":true}')
    client = LLMClient(provider='counter_test', model='test')
    def work():
        before = client.usage()['calls']
        for _ in range(3):
            client.complete_json('JSON')
        return client.usage()['calls'] - before
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: work(), range(2))) == [3, 3]
    assert client.total_usage()['calls'] == 6 and client.usage()['calls'] == 0


@pytest.mark.parametrize('timeout,retries', [(float('nan'), 1), (0, 1), (601, 1), (60, 4),
                                            (True, 1), (60, 1.5), (60, True)])
def test_invalid_sdk_limits_rejected(timeout, retries):
    with pytest.raises(LLMError):
        LLMClient(provider='fake', timeout=timeout, max_retries=retries)


def test_team_mode_cannot_silently_disable_access_checks(monkeypatch):
    monkeypatch.setenv('APP_ENV', 'team')
    monkeypatch.setenv('APP_AUTH_REQUIRED', '0')
    with pytest.raises(ConfigurationError, match='requires'):
        Settings.from_env()
    monkeypatch.setenv('APP_AUTH_REQUIRED', '1')
    monkeypatch.delenv('APP_ALLOWED_EMAILS', raising=False)
    with pytest.raises(ConfigurationError, match='APP_ALLOWED_EMAILS'):
        Settings.from_env()


def test_invalid_access_flag_is_rejected(monkeypatch):
    monkeypatch.setenv('APP_AUTH_REQUIRED', 'flase')
    with pytest.raises(ConfigurationError):
        Settings.from_env()


def test_team_identity_requires_verified_allowlisted_email():
    allowed = ('member@example.com',)
    assert authorized_identity({'email': 'Member@example.com', 'email_verified': True}, allowed)
    assert not authorized_identity({'email': 'outsider@example.com', 'email_verified': True}, allowed)
    assert not authorized_identity({'email': 'member@example.com', 'email_verified': False}, allowed)
    assert not authorized_identity({'email': 'member@example.com', 'email_verified': 'true'}, allowed)


def test_demo_sessions_share_store_and_do_not_lose_writes(tmp_path, monkeypatch):
    monkeypatch.setenv('APP_ENV', 'local')
    monkeypatch.setenv('APP_AUTH_REQUIRED', '0')
    with create_runtime(demo=True, data_path=tmp_path / 'state') as first, \
         create_runtime(demo=True, data_path=tmp_path / 'state') as second:
        assert first.graph_store is second.graph_store
        for runtime, name in ((first, 'a'), (second, 'b')):
            path = tmp_path / f'{name}.txt'
            path.write_text('Atlas is managed by Maya.', encoding='utf-8')
            assert runtime.ingestion.ingest_document(path)['status'] == 'ingested'
        assert len(first.graph_store.documents) == 2


def test_runtime_closes_database_when_vector_construction_fails(monkeypatch):
    import src.stores.neo4j_store as neo
    import src.stores.vector_store as vector
    monkeypatch.setenv('APP_ENV', 'local')
    monkeypatch.setenv('APP_AUTH_REQUIRED', '0')
    monkeypatch.setenv('LLM_PROVIDER', 'openai')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only')
    monkeypatch.setenv('LLM_FAKE', '0')
    graph = SimpleNamespace(connect=lambda: None, close=lambda: setattr(graph, 'closed', True), closed=False)
    monkeypatch.setattr(neo, 'Neo4jStore', lambda: graph)
    def fail(**kwargs):
        raise RuntimeError('vector unavailable')
    monkeypatch.setattr(vector, 'VectorStore', fail)
    with pytest.raises(RuntimeError):
        create_runtime()
    assert graph.closed


def test_team_app_stops_before_opening_workspace(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from src.runtime import ROOT
    monkeypatch.setenv('APP_ENV', 'team')
    monkeypatch.setenv('APP_AUTH_REQUIRED', '1')
    monkeypatch.setenv('APP_ALLOWED_EMAILS', 'member@example.com')
    app = AppTest.from_file(str(ROOT / 'app' / 'chatbot_app.py')).run()
    assert not app.exception
    assert any(b.label == 'Sign in' for b in app.button)
    assert not app.chat_input


def test_annotated_eval_reports_checks_and_latency(tmp_path):
    from eval.run_eval import run_eval
    cases = tmp_path / 'cases.jsonl'
    cases.write_text('{"question":"Who manages Atlas?","expected_terms":["Maya"],'
                     '"expected_doc_ids":["atlas"],"expect_insufficient":false}\n')
    query = SimpleNamespace(answer_question=lambda _: {
        'answer_text': 'Maya manages Atlas.', 'sources': [{'doc_id': 'atlas'}], 'insufficient_evidence': False})
    result = run_eval(cases, pipeline=query)
    assert result['annotated_cases'] == result['passed_annotated_cases'] == 1
    assert result['median_latency_seconds'] >= 0


def test_failed_annotated_case_stays_in_denominator(tmp_path):
    from eval.run_eval import run_eval
    cases = tmp_path / 'cases.jsonl'
    cases.write_text('{"question":"Who manages Atlas?","expected_terms":["Maya"]}\n')
    def fail(_):
        raise RuntimeError('service unavailable')
    result = run_eval(cases, pipeline=SimpleNamespace(answer_question=fail))
    assert result['annotated_cases'] == 1 and result['passed_annotated_cases'] == 0
    assert result['errors'] == 1
