"""Query -> retrieval -> reasoning -> critic -> cited response."""
from __future__ import annotations

import math
import re
from time import perf_counter

from src.query.query_agent import plan_query, MAX_QUESTION_CHARS
from src.query.retrieval_agent import retrieve
from src.reasoning.reasoning_agent import write_answer, NO_INFO_TEXT
from src.reasoning.critic_agent import review_answer, MAX_RETRIES


def _usage_snapshot(clients):
    """Optional client counters; scripted clients need not implement usage()."""
    snapshots = {}
    for client in clients:
        if client is None or id(client) in snapshots:
            continue
        usage = getattr(client, 'usage', None)
        if callable(usage):
            try:
                value = usage()
            except Exception:
                continue  # diagnostics must not turn a working answer into a failure
            if isinstance(value, dict):
                snapshots[id(client)] = (client, dict(value))
    return snapshots


def _usage_delta(before):
    totals = {}
    for client, initial in before.values():
        try:
            final = client.usage()
        except Exception:
            continue
        if not isinstance(final, dict):
            continue
        for key in ('calls', 'input_tokens', 'output_tokens', 'elapsed_seconds'):
            start, end = initial.get(key), final.get(key)
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                totals[key] = totals.get(key, 0) + max(0, end - start)
    return totals


class QueryPipeline:
    def __init__(self, graph_store, vector_store, *, query_llm=None, reasoning_llm=None,
                 critic_llm=None, top_k=5, graph_depth=2, min_score=0.25,
                 max_graph_facts=20, max_retries=MAX_RETRIES, retry_k_step=3):
        for name, value, lower, upper in (
            ('top_k', top_k, 1, 100), ('graph_depth', graph_depth, 1, 5),
            ('max_graph_facts', max_graph_facts, 1, 100),
            ('max_retries', max_retries, 0, MAX_RETRIES), ('retry_k_step', retry_k_step, 0, 100),
        ):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f'{name} must be an integer between {lower} and {upper}')
        if isinstance(min_score, bool) or not isinstance(min_score, (int, float)) \
                or not math.isfinite(min_score) or not 0 <= min_score <= 1:
            raise ValueError('min_score must be a finite number between 0 and 1')
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.query_llm = query_llm
        self.reasoning_llm = reasoning_llm
        self.critic_llm = critic_llm
        self.top_k = top_k
        self.graph_depth = graph_depth
        self.min_score = min_score
        self.max_graph_facts = max_graph_facts
        self.max_retries = max_retries
        self.retry_k_step = retry_k_step

    def answer_question(self, question, history=None):
        if not isinstance(question, str):
            raise TypeError('question must be a string')
        if not question.strip():
            raise ValueError('question is empty or has no searchable text')
        if len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f'question must be at most {MAX_QUESTION_CHARS} characters')
        started = perf_counter()
        timings = dict.fromkeys(('planning', 'retrieval', 'reasoning', 'critic'), 0.0)
        usage = _usage_snapshot((self.query_llm, self.reasoning_llm, self.critic_llm))
        stage = perf_counter()
        plan = plan_query(question, history, llm=self.query_llm)
        timings['planning'] += perf_counter() - stage
        retries = 0
        feedback = None
        while True:
            stage = perf_counter()
            # Only read the stores while holding the lock. A slow model response
            # must not block ingestion or another question. Capture provenance in
            # the same read so later document edits cannot change its source labels.
            with self.graph_store.lock:
                self.graph_store.refresh()
                evidence = retrieve(plan, min(100, self.top_k + retries * self.retry_k_step),
                                    graph_store=self.graph_store, vector_store=self.vector_store,
                                    graph_depth=self.graph_depth, min_score=self.min_score,
                                    max_graph_facts=self.max_graph_facts)
                source_notes = {}
                for item in evidence['chunks'] + evidence['graph_facts']:
                    ids = [item['note_id']] if 'note_id' in item else item['source_note_ids']
                    for nid in ids:
                        if nid not in source_notes:
                            source_notes[nid] = self.graph_store.get_note(nid)
            timings['retrieval'] += perf_counter() - stage
            stage = perf_counter()
            draft = write_answer(question, evidence, llm=self.reasoning_llm,
                                 normalized_question=plan['rewritten_query'], critic_feedback=feedback)
            timings['reasoning'] += perf_counter() - stage
            stage = perf_counter()
            decision = review_answer(question, draft, evidence, retries, llm=self.critic_llm)
            timings['critic'] += perf_counter() - stage
            if decision['action'] == 'accept' or retries >= self.max_retries:
                break
            retries += 1
            feedback = decision['reason']
            refined = decision['refined_query'] or plan['rewritten_query']
            key = ' '.join(refined.split()).casefold()
            existing = {' '.join(q.split()).casefold() for q in (question, plan['rewritten_query'])}
            if key not in existing:
                stage = perf_counter()
                plan = plan_query(decision['refined_query'], history, llm=self.query_llm)
                timings['planning'] += perf_counter() - stage
            plan = {**plan, 'attempt': retries, 'route': 'hybrid'}
        valid = decision['supported']
        sources = {}
        cited = (draft.cited_ids() | set(re.findall(r'\[([A-Za-z]\d+)\]', draft.answer_text))) if valid else set()
        evidence_items = []
        for item in evidence['chunks'] + evidence['graph_facts']:
            if item['evidence_id'] not in cited:
                continue
            evidence_items.append(item)
            note_ids = [item['note_id']] if 'note_id' in item else item['source_note_ids']
            for nid in note_ids:
                note = source_notes.get(nid)
                if note is None:
                    continue
                source = sources.setdefault(nid, {k: note.get(k) for k in
                    ('doc_id', 'section', 'note_id', 'source_filename', 'page')})
                ids = source.setdefault('evidence_ids', [])
                if item['evidence_id'] not in ids:
                    ids.append(item['evidence_id'])
        warnings = list(draft.warnings)
        if not valid:
            warnings.append(decision['reason'])
        if evidence.get('notes'):
            warnings.append(evidence['notes'])
        timings['total'] = perf_counter() - started
        return {'answer_text': draft.answer_text if valid else NO_INFO_TEXT,
                'sources': list(sources.values()), 'confidence': draft.confidence if valid else 0.0,
                'retries_used': retries, 'insufficient_evidence': not (valid and draft.evidence_sufficient),
                'state': draft.state.value if draft.state and valid else 'INSUFFICIENT',
                'warnings': list(dict.fromkeys(warnings)), 'evidence': evidence_items,
                'timings_seconds': {k: round(v, 6) for k, v in timings.items()},
                'llm_usage': _usage_delta(usage)}


def answer_question(question, history=None, *, pipeline=None):
    if pipeline is None:
        from src.runtime import create_runtime
        with create_runtime() as runtime:
            return runtime.query.answer_question(question, history)
    return pipeline.answer_question(question, history)
