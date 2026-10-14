from copy import deepcopy

from src.demo import DemoLLM, LocalSearchStore
from src.ingestion.extraction_agent import ExtractionAgent
from src.ingestion.graph_agent import GraphAgent, MemoryStore
from src.ingestion.pipeline import IngestionPipeline
from src.pipeline import QueryPipeline
from src.query.retrieval_agent import retrieve
from src.reasoning.critic_agent import review_answer
from src.runtime import create_runtime
from test_graph_agent import note, DecisionLLM
from fixtures_reasoning import bundle, chunk, good_draft


def test_document_to_cited_answer_and_reopen(tmp_path):
    path = tmp_path / 'atlas.txt'
    path.write_text('Atlas is managed by Maya.', encoding='utf-8')
    with create_runtime(demo=True, data_path=tmp_path / 'state') as runtime:
        report = runtime.ingestion.ingest_document(path)
        assert report['status'] == 'ingested' and report['vector_pending'] == 0
        assert len(runtime.graph_store.edges) == 1
        duplicate = runtime.ingestion.ingest_document(path)
        assert duplicate['status'] == 'skipped_duplicate'
        result = runtime.query.answer_question('Who manages Atlas?')
        assert 'Maya' in result['answer_text'] and not result['insufficient_evidence']
        assert result['sources'][0]['doc_id'] == 'atlas'
        assert result['sources'][0]['source_filename'] == 'atlas.txt'
    with create_runtime(demo=True, data_path=tmp_path / 'state') as runtime:
        result = runtime.query.answer_question('Who manages Atlas?')
        assert result['sources'] and 'Maya' in result['answer_text']


def test_two_hop_answer_maps_fact_citations_to_original_document():
    from test_fake_reasoning import FakeLLM, analysis, answer, claim
    text = 'Acme built Atlas. Acme is led by Maya.'
    extraction = {'is_informative': True, 'entities': [
        {'name': 'Atlas', 'type': 'Product'}, {'name': 'Acme', 'type': 'Organization'},
        {'name': 'Maya', 'type': 'Person'}], 'relations': [
        {'subject': 'Acme', 'relation': 'built', 'object': 'Atlas',
         'evidence_quote': 'Acme built Atlas.', 'event_time': None},
        {'subject': 'Acme', 'relation': 'led_by', 'object': 'Maya',
         'evidence_quote': 'Acme is led by Maya.', 'event_time': None}],
        'keywords': ['Atlas'], 'tags': [], 'context': text, 'title': 'Leadership'}
    graph, vector = MemoryStore(), LocalSearchStore()
    extracted = ExtractionAgent(FakeLLM([extraction])).extract(text, 'leadership')
    GraphAgent(graph, vector).ingest_note(extracted)
    question = 'Who leads the organization that built Atlas?'
    planner = FakeLLM([{'rewritten_query': question, 'route': 'graph', 'target_entities': ['Atlas']}])
    reasoner = FakeLLM([analysis(question), answer(
        [claim('C1', 'Maya leads the organization that built Atlas', 'GRAPH', ['F1', 'F2'], 'derived')],
        'Maya leads Acme, which built Atlas [F1] [F2].',
        steps=[{'step': 1, 'description': 'Acme built Atlas and Acme is led by Maya.',
                'evidence_ids': ['F1', 'F2']}])])
    query = QueryPipeline(graph, vector, query_llm=planner, reasoning_llm=reasoner,
                          critic_llm=DemoLLM('critic'))
    result = query.answer_question(question)
    assert not result['insufficient_evidence'] and result['retries_used'] == 0
    assert result['sources'][0]['doc_id'] == 'leadership'
    assert set(result['sources'][0]['evidence_ids']) == {'F1', 'F2'}


def test_empty_corpus_stops_after_two_retries(tmp_path):
    with create_runtime(demo=True, data_path=tmp_path) as runtime:
        result = runtime.query.answer_question('Who manages Atlas?')
        assert result['retries_used'] == 2
        assert result['insufficient_evidence'] and not result['sources']
        assert result['confidence'] == 0


def test_replacement_removes_old_evidence(tmp_path):
    path = tmp_path / 'atlas.txt'
    path.write_text('Atlas is managed by Maya.', encoding='utf-8')
    with create_runtime(demo=True, data_path=tmp_path / 'state') as runtime:
        runtime.ingestion.ingest_document(path)
        path.write_text('Atlas is managed by Arun.', encoding='utf-8')
        assert runtime.ingestion.ingest_document(path)['status'] == 'failed'
        assert runtime.ingestion.ingest_document(path, replace=True)['status'] == 'replaced'
        result = runtime.query.answer_question('Who manages Atlas?')
        assert 'Arun' in result['answer_text'] and 'Maya' not in result['answer_text']
        assert len(runtime.vector_store.records) == 1


def test_failed_replacement_preserves_previous_version(tmp_path):
    path = tmp_path / 'atlas.txt'
    path.write_text('Atlas is managed by Maya.', encoding='utf-8')
    with create_runtime(demo=True, data_path=tmp_path / 'state') as runtime:
        runtime.ingestion.ingest_document(path)
        before = runtime.graph_store.snapshot()
        path.write_text('Atlas is managed by Arun.', encoding='utf-8')
        class Broken:
            def extract_note(self, chunk):
                raise ValueError('bad output')
        runtime.ingestion.extractor = Broken()
        assert runtime.ingestion.ingest_document(path, replace=True)['status'] == 'failed'
        assert runtime.graph_store.snapshot() == before


def test_replacement_commit_failure_is_atomic(tmp_path):
    path = tmp_path / 'atlas.txt'
    path.write_text('Atlas is managed by Maya.', encoding='utf-8')
    with create_runtime(demo=True, data_path=tmp_path / 'state') as runtime:
        runtime.ingestion.ingest_document(path)
        before = runtime.graph_store.snapshot()
        vectors = deepcopy(runtime.vector_store.records)
        path.write_text('Atlas is managed by Arun.', encoding='utf-8')
        def fail():
            raise RuntimeError('database offline')
        runtime.graph_store.commit = fail
        assert runtime.ingestion.ingest_document(path, replace=True)['status'] == 'failed'
        assert runtime.graph_store.snapshot() == before and runtime.vector_store.records == vectors


def test_partial_ingestion_can_be_retried(tmp_path, monkeypatch):
    chunks = [{'text': 'Atlas is managed by Maya.', 'doc_id': 'atlas', 'chunk_index': i,
               'content_hash': 'abc'} for i in range(2)]
    from src.ingestion.parser.schema import Chunk
    chunks = [Chunk(**c, source_filename='atlas.txt', parser_version='test') for c in chunks]
    monkeypatch.setattr('src.ingestion.pipeline.parse_document', lambda _: chunks)
    path = tmp_path / 'atlas.txt'
    path.write_text('Atlas is managed by Maya.', encoding='utf-8')
    from src.ingestion.parser.validator import file_hash
    for c in chunks:
        c.content_hash = file_hash(path)
    extractor = ExtractionAgent(DemoLLM('extraction'))
    original = extractor.extract_note
    def sometimes(chunk):
        if chunk.chunk_index == 1:
            raise ValueError('temporary failure')
        return original(chunk)
    extractor.extract_note = sometimes
    pipeline = IngestionPipeline(extractor, GraphAgent(MemoryStore(), LocalSearchStore()))
    result = pipeline.ingest_document(path)
    assert result['status'] == 'failed' and len(pipeline.graph_agent.graph_store.notes) == 1
    extractor.extract_note = original
    assert pipeline.ingest_document(path)['status'] == 'ingested'
    assert len(pipeline.graph_agent.graph_store.notes) == 2


def test_index_failure_reports_pending_and_recovers(tmp_path):
    path = tmp_path / 'atlas.txt'
    path.write_text('Atlas is managed by Maya.', encoding='utf-8')
    vector = LocalSearchStore()
    original = vector.add
    original_batch = vector.add_many
    def fail(*args, **kwargs):
        raise RuntimeError('index offline')
    vector.add = fail
    vector.add_many = fail
    agent = GraphAgent(MemoryStore(), vector)
    pipeline = IngestionPipeline(ExtractionAgent(DemoLLM('extraction')), agent)
    result = pipeline.ingest_document(path)
    assert result['status'] == 'ingested' and result['vector_pending'] == 1 and result['index_errors']
    vector.add = original
    vector.add_many = original_batch
    result = pipeline.ingest_document(path)
    assert result['status'] == 'skipped_duplicate' and result['vector_pending'] == 0


def test_superseded_memory_is_filtered_in_both_sources():
    graph, vector = MemoryStore(), LocalSearchStore()
    agent = GraphAgent(graph, vector)
    first = note()
    agent.ingest_note(first)
    agent.llm = DecisionLLM(graph, 'update')
    second = note('new', '2026-10-01T00:00:00Z', 'Arun')
    agent.ingest_note(second)
    plan = {'original_question': 'Atlas', 'rewritten_query': 'Atlas', 'route': 'hybrid',
            'target_entities': ['Atlas'], 'attempt': 0}
    evidence = retrieve(plan, graph_store=graph, vector_store=vector)
    assert [c['note_id'] for c in evidence['chunks']] == [second.note_id]
    assert [f['object'] for f in evidence['graph_facts']] == ['Arun']
    assert not graph.is_note_active(first.note_id)


def test_shared_entity_and_memory_survive_document_removal():
    graph = MemoryStore()
    agent = GraphAgent(graph)
    agent.ingest_note(note('first'))
    agent.ingest_note(note('second'))
    graph.delete_document('first')
    assert len(graph.notes) == 1 and len(graph.memories) == 1
    facts = graph.query_subgraph(['Atlas'])['edges']
    assert len(facts) == 1 and len(facts[0]['source_note_ids']) == 1


def test_critic_rejects_unknown_citations_even_at_cap():
    draft = good_draft()
    draft.answer_text += ' [E999]'
    evidence = bundle([chunk('E1', 'OpenAI released GPT-4 in March 2023.')])
    decision = review_answer('What was released?', draft, evidence, 2)
    assert decision['action'] == 'accept' and not decision['supported']


def test_critic_checks_semantic_support():
    class Reject:
        def complete_json(self, *args, **kwargs):
            return {'supported': False, 'reason': 'Claim is unrelated to evidence'}
    draft = good_draft()
    evidence = bundle([chunk('E1', 'OpenAI released GPT-4 in March 2023.')])
    decision = review_answer('What was released?', draft, evidence, 0, llm=Reject())
    assert decision['action'] == 'retry' and not decision['supported']


def test_semantic_rejection_cannot_escape_retry_cap(tmp_path):
    path = tmp_path / 'atlas.txt'
    path.write_text('Atlas is managed by Maya.', encoding='utf-8')
    class Reject:
        def complete_json(self, *args, **kwargs):
            return {'supported': False, 'reason': 'Unsupported answer'}
    with create_runtime(demo=True, data_path=tmp_path / 'state') as runtime:
        runtime.ingestion.ingest_document(path)
        runtime.query.critic_llm = Reject()
        result = runtime.query.answer_question('Who manages Atlas?')
        assert result['insufficient_evidence'] and result['retries_used'] == 2
        assert not result['sources'] and 'Maya' not in result['answer_text']


def test_eval_reads_questions_and_executes(tmp_path):
    from eval.run_eval import run_eval
    path = tmp_path / 'questions.md'
    path.write_text('# Questions\nInstructions, not a question.\n1. Who manages Atlas?\n2. Any evidence?\n')
    with create_runtime(demo=True, data_path=tmp_path / 'state') as runtime:
        result = run_eval(path, pipeline=runtime.query)
    assert result['total_questions'] == 2 and len(result['results']) == 2 and result['errors'] == 0
