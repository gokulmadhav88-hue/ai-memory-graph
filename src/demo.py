"""Offline UI demonstration. Uses extractive fixtures and keyword search, not an AI model."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import html
import json
import os
import re

STOP = {'who', 'what', 'when', 'where', 'how', 'the', 'a', 'an', 'is', 'are', 'was',
        'by', 'of', 'in', 'to', 'and', 'does', 'do', 'about', 'it', 'that', 'which'}


def tokens(text):
    return {w for w in re.findall(r'\w+', text.casefold()) if w not in STOP}


class LocalSearchStore:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.records = json.loads(self.path.read_text(encoding='utf-8')) if self.path and self.path.exists() else {}

    def _commit(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps(self.records), encoding='utf-8')
            os.replace(temporary, self.path)

    def add(self, note_id, text, metadata):
        self.records[note_id] = {'note_id': note_id, 'text': text, 'metadata': deepcopy(metadata)}
        self._commit()

    def add_many(self, records):
        for record in records:
            self.records[record['note_id']] = {'note_id': record['note_id'], 'text': record['text'],
                                              'metadata': deepcopy(record['metadata'])}
        self._commit()

    def delete_notes(self, note_ids):
        removed = sum(nid in self.records for nid in set(note_ids))
        for nid in note_ids:
            self.records.pop(nid, None)
        self._commit()
        return removed

    def query(self, query_text, top_k=5):
        query = tokens(query_text)
        results = [dict(r, score=len(query & tokens(r['text'])) / max(1, len(query)))
                   for r in self.records.values()]
        return sorted((r for r in results if r['score'] > 0), key=lambda r: r['score'], reverse=True)[:top_k]

    def delete_document(self, doc_id):
        ids = [nid for nid, r in self.records.items() if r['metadata']['doc_id'] == doc_id]
        for nid in ids:
            del self.records[nid]
        self._commit()
        return len(ids)


class DemoLLM:
    def __init__(self, agent):
        self.agent = agent

    def complete_json(self, prompt, system=None, max_tokens=1000):
        if self.agent == 'extraction':
            data = json.loads(prompt.split('\nRepair error:')[0])
            text = data['text']
            entities, relations = {}, []
            for match in re.finditer(r'([A-Z][\w-]*) is managed by ([A-Z][\w-]*)\.', text):
                subject, obj = match.group(1, 2)
                entities[subject] = {'name': subject, 'type': 'Product'}
                entities[obj] = {'name': obj, 'type': 'Person'}
                relations.append({'subject': subject, 'relation': 'managed_by', 'object': obj,
                                  'evidence_quote': match.group(), 'event_time': None})
            informative = len(text.strip()) > 15
            return {'is_informative': informative, 'entities': list(entities.values()),
                    'relations': relations, 'keywords': sorted(tokens(text))[:8] if informative else [],
                    'tags': ['demo'] if informative else [], 'context': text[:180] if informative else '',
                    'title': 'Demo document'}
        if self.agent == 'query':
            question = re.search(r'<question>\s*(.*?)\s*</question>', prompt, re.S).group(1)
            return {'rewritten_query': question, 'route': 'hybrid',
                    'target_entities': re.findall(r'\b[A-Z][\w-]+\b', question)}
        if self.agent == 'graph':
            data = json.loads(prompt)
            if 'candidates' in data:
                return {'node_id': None}
            return {'action': 'add', 'target_ids': [], 'reason': 'Preserve demo evidence'}
        if self.agent == 'critic':
            return {'supported': True, 'reason': 'Offline demo quotes the retrieved passage.'}
        if prompt.startswith('[TASK:ANALYSIS]'):
            question = re.search(r'Question: (.*)', prompt).group(1)
            return {'normalized_question': question, 'question_types': ['single_hop'], 'entities': [],
                    'sub_questions': [{'text': question}], 'needs_temporal': False, 'ambiguity_notes': [],
                    'false_premise_notes': [], 'assumptions': []}
        items = re.findall(r'<evidence id="([^"]+)"[^>]*>(.*?)</evidence>', prompt, re.S)
        if not items:
            raise ValueError('Demo expected retrieved evidence')
        eid, text = next((item for item in items if item[0].startswith('E')), items[0])
        text = html.unescape(text).strip()
        question = re.search(r'Question as asked: (.*)', prompt).group(1)
        return {'sub_questions': [{'text': question, 'status': 'answered'}], 'triage': [], 'claims': [
                    {'claim_id': 'C1', 'text': text, 'source': 'GRAPH',
                     'kind': 'explicit', 'evidence_ids': [eid]}],
                'reasoning_steps': [], 'conflicts': [], 'missing_items': [], 'assumptions': [],
                'verify_topics': [], 'answer_text': f'From the uploaded document: "{text}" [{eid}].'}
