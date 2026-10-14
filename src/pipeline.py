"""Query -> retrieval -> reasoning -> critic -> cited response."""
from __future__ import annotations

from src.query.query_agent import plan_query
from src.query.retrieval_agent import retrieve
from src.reasoning.reasoning_agent import write_answer, NO_INFO_TEXT
from src.reasoning.critic_agent import review_answer


class QueryPipeline:
    def __init__(self, graph_store, vector_store, *, query_llm=None, reasoning_llm=None,
                 critic_llm=None, top_k=5):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.query_llm = query_llm
        self.reasoning_llm = reasoning_llm
        self.critic_llm = critic_llm
        self.top_k = top_k

    def answer_question(self, question, history=None):
        with self.graph_store.lock:
            self.graph_store.refresh()
            plan = plan_query(question, history, llm=self.query_llm)
            retries = 0
            feedback = None
            while True:
                evidence = retrieve(plan, self.top_k + retries * 3, graph_store=self.graph_store,
                                    vector_store=self.vector_store)
                draft = write_answer(question, evidence, llm=self.reasoning_llm,
                                     normalized_question=plan['rewritten_query'], critic_feedback=feedback)
                decision = review_answer(question, draft, evidence, retries, llm=self.critic_llm)
                if decision['action'] == 'accept':
                    break
                retries += 1
                feedback = decision['reason']
                plan = plan_query(decision['refined_query'], history, llm=self.query_llm)
                plan['attempt'] = retries
                # Retry both sources, even when the initial question chose graph only.
                plan['route'] = 'hybrid'
            valid = decision['supported']
            sources = {}
            cited = set(draft.evidence_used) if valid else set()
            evidence_items = []
            for item in evidence['chunks'] + evidence['graph_facts']:
                if item['evidence_id'] not in cited:
                    continue
                evidence_items.append(item)
                note_ids = [item['note_id']] if 'note_id' in item else item['source_note_ids']
                for nid in note_ids:
                    note = self.graph_store.get_note(nid)
                    if note is None:
                        continue
                    source = sources.setdefault(nid, {k: note.get(k) for k in
                        ('doc_id', 'section', 'note_id', 'source_filename', 'page')})
                    source.setdefault('evidence_ids', []).append(item['evidence_id'])
            warnings = list(draft.warnings)
            if not valid:
                warnings.append(decision['reason'])
            if evidence.get('notes'):
                warnings.append(evidence['notes'])
            return {'answer_text': draft.answer_text if valid else NO_INFO_TEXT,
                    'sources': list(sources.values()), 'confidence': draft.confidence if valid else 0.0,
                    'retries_used': retries, 'insufficient_evidence': not (valid and draft.evidence_sufficient),
                    'state': draft.state.value if draft.state else None, 'warnings': warnings,
                    'evidence': evidence_items}


def answer_question(question, history=None, *, pipeline=None):
    if pipeline is None:
        from src.runtime import create_runtime
        with create_runtime() as runtime:
            return runtime.query.answer_question(question, history)
    return pipeline.answer_question(question, history)
