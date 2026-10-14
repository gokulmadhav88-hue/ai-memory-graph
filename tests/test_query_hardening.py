"""Regression checks for bounded retrieval, provenance and query concurrency."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from threading import Event

import pytest

from src.demo import DemoLLM, LocalSearchStore
from src.ingestion.graph_agent import GraphAgent, MemoryStore
from src.pipeline import QueryPipeline
from src.query.query_agent import plan_query, _clean_history
from src.query.retrieval_agent import retrieve
from src.reasoning.critic_agent import review_answer
from src.reasoning.schemas import MissingItem
from fixtures_reasoning import bundle, chunk, good_draft
from test_graph_agent import note
from test_query_agent import llm as query_llm
from test_retrieval_agent import FakeNeo4jStore, FakeVectorStore, GPT4_SUBGRAPH, GOOD, plan, vres


class SourcedGraph(FakeNeo4jStore):
    def __init__(self):
        super().__init__(deepcopy(GPT4_SUBGRAPH))
        self.notes = {
            'n1': {'note_id': 'n1', 'doc_id': 'original', 'section': 'Leadership',
                   'content': 'OpenAI released GPT-4. Sam Altman was CEO at the time.'},
            'n2': {'note_id': 'n2', 'doc_id': 'other', 'content': 'Other relevant information.'},
        }
        self.active = {'n1', 'n2'}
        self.closed = False

    def get_note(self, nid):
        return deepcopy(self.notes.get(nid))

    def is_note_active(self, nid):
        return nid in self.active

    def close(self):
        self.closed = True


@pytest.mark.parametrize('score', [float('nan'), float('inf'), float('-inf'), 'NaN', True])
def test_nonfinite_or_boolean_scores_cannot_become_high_confidence_hits(score):
    result = retrieve(plan('vector', ()), vector_store=FakeVectorStore(
        [vres('bad', 'Bad hit', score), GOOD[0]]))
    assert [c['note_id'] for c in result['chunks']] == ['n1']
    assert result['was_filtered']


@pytest.mark.parametrize('kwargs', [
    {'min_score': float('nan')}, {'min_score': True}, {'min_score': 1.5},
    {'graph_depth': 0}, {'graph_depth': 6}, {'max_graph_facts': 0},
])
def test_invalid_retrieval_configuration_fails_before_store_calls(kwargs):
    vector = FakeVectorStore(GOOD)
    with pytest.raises(ValueError):
        retrieve(plan('vector', ()), vector_store=vector, **kwargs)
    assert vector.calls == []


def test_retrieval_depth_and_fact_budget_are_configurable():
    graph = FakeNeo4jStore(GPT4_SUBGRAPH)
    result = retrieve(plan(), graph_store=graph, vector_store=FakeVectorStore(),
                      graph_depth=1, max_graph_facts=1)
    assert graph.calls == [(['GPT-4'], 1)]
    assert len(result['graph_facts']) == 1
    assert result['graph_facts'][0]['relation'] == 'released'
    assert result['was_filtered']


def test_unmatched_graph_target_cannot_return_unrelated_facts():
    result = retrieve(plan(entities=['Missing Entity']), graph_store=FakeNeo4jStore(GPT4_SUBGRAPH),
                      vector_store=FakeVectorStore())
    assert result['graph_facts'] == []
    assert result['was_filtered']


def test_graph_source_passage_survives_full_vector_result_budget():
    graph = SourcedGraph()
    result = retrieve(plan('hybrid'), top_k=1, graph_store=graph,
                      vector_store=FakeVectorStore([GOOD[1]]))
    assert result['chunks'][0]['note_id'] == 'n1'
    assert 'at the time' in result['chunks'][0]['text']
    assert result['chunks'][0]['source'] == 'graph'
    assert result['was_filtered']


def test_stale_fact_sources_are_removed_even_if_adapter_returns_stale_edges():
    graph = SourcedGraph()
    graph.active = {'n2'}
    result = retrieve(plan('hybrid'), graph_store=graph, vector_store=FakeVectorStore(GOOD))
    assert result['graph_facts'] == []
    assert [c['note_id'] for c in result['chunks']] == ['n2']


def test_partial_graph_source_validation_failure_discards_all_unchecked_facts():
    graph = SourcedGraph()
    def fail(nid):
        raise RuntimeError('source read unavailable')
    graph.get_note = fail
    result = retrieve(plan('hybrid'), graph_store=graph, vector_store=FakeVectorStore(GOOD))
    assert result['graph_facts'] == []
    assert len(result['chunks']) == 2
    assert 'graph query failed' in result['notes']


def test_implicitly_opened_graph_driver_is_closed(monkeypatch):
    graph = SourcedGraph()
    monkeypatch.setattr('src.query.retrieval_agent._default_graph_store', lambda: graph)
    result = retrieve(plan(), vector_store=FakeVectorStore())
    assert graph.closed and result['chunks']


def test_vector_initialization_failure_keeps_successful_graph_evidence(monkeypatch):
    def fail():
        raise RuntimeError('vector collection unavailable')
    monkeypatch.setattr('src.query.retrieval_agent._vector_store_module.get_default_store', fail)
    result = retrieve(plan('hybrid'), graph_store=FakeNeo4jStore(GPT4_SUBGRAPH))
    assert result['graph_facts'] and 'vector query failed' in result['notes']


def test_entity_boundaries_prevent_partial_name_matches():
    question = 'Tell me about Atlassian.'
    result = plan_query(question, llm=query_llm(question, 'hybrid', ['Atlas', 'Atlassian']))
    assert result['target_entities'] == ['Atlassian']
    result = plan_query('Who made gpt4?', llm=query_llm('Who made GPT-4?', 'graph', ['GPT-4']))
    assert result['target_entities'] == ['GPT-4']


def test_history_retains_recent_valid_turns_in_original_order():
    history = [{'role': 'user', 'content': f'Turn {i}'} for i in range(100)] + [None]
    assert [t['content'] for t in _clean_history(history)] == [f'Turn {i}' for i in range(94, 100)]


def test_critic_can_see_uncited_contradictions():
    class Audit:
        payload = None
        def complete_json(self, prompt, **kwargs):
            self.payload = json.loads(prompt)
            return {'supported': False, 'reason': 'Uncited evidence conflicts with the answer.'}
    audit = Audit()
    evidence = bundle([chunk('E1', 'OpenAI released GPT-4 in March 2023.'),
                       chunk('E2', 'The release occurred in April 2023 instead.')])
    result = review_answer('When was it released?', good_draft(), evidence, 0, llm=audit)
    assert {i['evidence_id'] for i in audit.payload['evidence']} == {'E1', 'E2'}
    assert not result['supported']


def test_critic_prompt_evidence_budget_and_citation_priority():
    class Audit:
        payload = None
        def complete_json(self, prompt, **kwargs):
            self.payload = json.loads(prompt)
            return {'supported': True, 'reason': 'Citations support the answer.'}
    audit = Audit()
    evidence = bundle([chunk(f'E{i}', 'Unrelated text. ' * 300) for i in range(2, 40)] +
                      [chunk('E1', 'OpenAI released GPT-4 in March 2023.')])
    result = review_answer('When was it released?', good_draft(), evidence, 0, llm=audit)
    items = audit.payload['evidence']
    assert result['supported'] and items[0]['evidence_id'] == 'E1'
    assert len(items) <= 30 and sum(len(i['text']) for i in items) <= 24000
    assert audit.payload['evidence_omitted_ids']
    assert any(i['text_truncated'] for i in items)


def _demo_pipeline():
    graph, vector = MemoryStore(), LocalSearchStore()
    GraphAgent(graph, vector).ingest_note(note())
    return QueryPipeline(graph, vector, query_llm=DemoLLM('query'),
                         reasoning_llm=DemoLLM('reasoning'), critic_llm=DemoLLM('critic'))


@pytest.mark.parametrize('stage', ['plan_query', 'write_answer', 'review_answer'])
def test_model_steps_do_not_hold_store_lock(monkeypatch, stage):
    import src.pipeline as module
    pipeline = _demo_pipeline()
    original = getattr(module, stage)
    entered, release = Event(), Event()
    def block(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, stage, block)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(pipeline.answer_question, 'Who manages Atlas?')
        try:
            assert entered.wait(5)
            acquired = pipeline.graph_store.lock.acquire(blocking=False)
            if acquired:
                pipeline.graph_store.lock.release()
            assert acquired, f'{stage} blocked store access while waiting for a model'
        finally:
            release.set()
        assert not future.result(timeout=5)['insufficient_evidence']


def test_empty_corpus_plans_once_and_uses_no_reasoning_or_critic_calls():
    class Counting(DemoLLM):
        def __init__(self, agent):
            super().__init__(agent)
            self.calls = 0
        def complete_json(self, *args, **kwargs):
            self.calls += 1
            return super().complete_json(*args, **kwargs)
    clients = {name: Counting(name) for name in ('query', 'reasoning', 'critic')}
    pipeline = QueryPipeline(MemoryStore(), LocalSearchStore(), query_llm=clients['query'],
                             reasoning_llm=clients['reasoning'], critic_llm=clients['critic'])
    result = pipeline.answer_question('Who manages Atlas?')
    assert result['retries_used'] == 2 and result['insufficient_evidence']
    assert {name: client.calls for name, client in clients.items()} == {
        'query': 1, 'reasoning': 0, 'critic': 0}


def test_pipeline_enforces_its_own_retry_cap(monkeypatch):
    monkeypatch.setattr('src.pipeline.review_answer', lambda *args, **kwargs: {
        'action': 'retry', 'refined_query': 'Who manages Atlas?',
        'reason': 'Unsupported answer.', 'supported': False})
    result = _demo_pipeline().answer_question('Who manages Atlas?')
    assert result['retries_used'] == 2 and result['insufficient_evidence']
    assert result['state'] == 'INSUFFICIENT' and not result['sources']
    assert 'Maya' not in result['answer_text']


def test_provenance_is_captured_before_concurrent_document_removal(monkeypatch):
    import src.pipeline as module
    pipeline = _demo_pipeline()
    original = module.write_answer
    doc_id = next(iter(pipeline.graph_store.notes.values()))['doc_id']
    def delete_during_reasoning(*args, **kwargs):
        with pipeline.graph_store.lock:
            pipeline.graph_store.delete_document(doc_id)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'write_answer', delete_during_reasoning)
    result = pipeline.answer_question('Who manages Atlas?')
    assert result['sources'] and result['sources'][0]['doc_id'] == doc_id
    assert not pipeline.graph_store.notes


def test_response_stage_timings_are_consistent():
    result = _demo_pipeline().answer_question('Who manages Atlas?')
    times = result['timings_seconds']
    assert set(times) == {'planning', 'retrieval', 'reasoning', 'critic', 'total'}
    assert all(isinstance(value, (int, float)) and value >= 0 for value in times.values())
    assert times['total'] + 0.000005 >= sum(value for key, value in times.items() if key != 'total')


def test_response_usage_reports_current_request_rather_than_lifetime_counts():
    class Metered(DemoLLM):
        def __init__(self, agent):
            super().__init__(agent)
            self.calls = 20
        def usage(self):
            return {'calls': self.calls, 'input_tokens': self.calls * 7,
                    'output_tokens': self.calls * 3, 'elapsed_seconds': self.calls / 10}
        def complete_json(self, *args, **kwargs):
            self.calls += 1
            return super().complete_json(*args, **kwargs)
    pipeline = _demo_pipeline()
    pipeline.query_llm = Metered('query')
    pipeline.reasoning_llm = Metered('reasoning')
    pipeline.critic_llm = Metered('critic')
    first = pipeline.answer_question('Who manages Atlas?')['llm_usage']
    second = pipeline.answer_question('Who manages Atlas?')['llm_usage']
    for counts in (first, second):
        assert counts['calls'] == 4
        assert counts['input_tokens'] == 28 and counts['output_tokens'] == 12
        assert counts['elapsed_seconds'] == pytest.approx(0.4)


def test_optional_usage_instrumentation_failure_does_not_break_answers():
    class BrokenMeter(DemoLLM):
        def usage(self):
            raise RuntimeError('metrics unavailable')
    pipeline = _demo_pipeline()
    pipeline.query_llm = BrokenMeter('query')
    result = pipeline.answer_question('Who manages Atlas?')
    assert not result['insufficient_evidence'] and result['llm_usage'] == {}


def test_retry_budget_can_be_disabled_without_accepting_unsupported_answers():
    pipeline = _demo_pipeline()
    pipeline.max_retries = 0
    class Reject:
        def complete_json(self, *args, **kwargs):
            return {'supported': False, 'reason': 'The answer is unsupported.'}
    pipeline.critic_llm = Reject()
    result = pipeline.answer_question('Who manages Atlas?')
    assert result['retries_used'] == 0 and result['insufficient_evidence']
    assert result['confidence'] == 0 and not result['sources']


def test_critic_refinement_uses_missing_subquestion_without_duplicating_original():
    question = 'Who manages Atlas?'
    draft = good_draft(evidence_sufficient=False, state='INSUFFICIENT', missing_info='A date is missing.')
    # The schema classes can be loaded under either package spelling by legacy tests.
    draft.missing_items = [MissingItem(question, 'No usable evidence'),
                           MissingItem('When did the manager change?', 'A date is missing.')]
    result = review_answer(question, draft, bundle([chunk('E1', 'OpenAI released GPT-4 in March 2023.')]), 0)
    assert result['refined_query'] == 'Who manages Atlas? When did the manager change?'


@pytest.mark.parametrize('kwargs', [{'top_k': 0}, {'max_retries': 3}, {'max_retries': True},
                                  {'graph_depth': 7}, {'min_score': float('inf')}])
def test_pipeline_validates_configuration(kwargs):
    with pytest.raises(ValueError):
        QueryPipeline(MemoryStore(), LocalSearchStore(), **kwargs)


@pytest.mark.parametrize('question,error', [(None, TypeError), ('   ', ValueError),
                                          ('x' * 2001, ValueError)])
def test_bad_request_is_rejected_before_any_model_or_store_call(question, error):
    graph = MemoryStore()
    def unexpected(*args, **kwargs):
        pytest.fail('an invalid request must not call a model or read stores')
    graph.refresh = unexpected
    class UnusedLLM:
        usage = unexpected
        complete_json = unexpected
    pipeline = QueryPipeline(graph, LocalSearchStore(), query_llm=UnusedLLM())
    with pytest.raises(error):
        pipeline.answer_question(question)
