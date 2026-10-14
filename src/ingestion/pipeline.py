"""Connect parser, extractor, graph writer, and vector outbox."""
from __future__ import annotations

from pathlib import Path

from .parser import parse_document
from .parser.validator import file_hash, validate_file


class IngestionPipeline:
    def __init__(self, extractor, graph_agent):
        self.extractor = extractor
        self.graph_agent = graph_agent

    def ingest_document(self, file_path, replace=False):
        path = Path(file_path)
        report = {'doc_id': path.stem, 'content_hash': '', 'status': 'failed',
                  'chunks_total': 0, 'notes_informative': 0, 'notes_skipped': 0,
                  'nodes_created': 0, 'nodes_merged': 0, 'edges_created': 0,
                  'errors': [], 'vector_pending': 0, 'vector_cleanup_pending': 0, 'index_errors': []}
        store = self.graph_agent.graph_store
        try:
            digest = file_hash(validate_file(path))
            report['content_hash'] = digest
            doc_id = path.stem
            with store.lock:
                store.refresh()
                if store.document_exists(doc_id, digest):
                    report['status'] = 'skipped_duplicate'
                    return self._outbox(report)
                replacing = self._is_replacement(store, doc_id, digest)
                if replacing and not replace:
                    report['errors'].append('A different file with this document ID exists. Enable replace to update it.')
                    return report
            # Parsing and extraction can make slow model/network calls. Readers should
            # still be able to use the committed corpus while those calls run.
            chunks = parse_document(path)
            report['chunks_total'] = len(chunks)
            if not chunks:
                report['errors'].append('No readable text found. The existing document was kept.')
                return report
            if file_hash(path) != digest:
                report['errors'].append('The file changed while being parsed. Retry with a stable file.')
                return report
            notes = []
            for chunk in chunks:
                try:
                    notes.append(self.extractor.extract_note(chunk))
                except Exception as exc:
                    report['errors'].append(f'Chunk {chunk.chunk_index}: {exc}')
            with store.lock:
                # Another upload may have committed while extraction was running.
                store.refresh()
                if store.document_exists(doc_id, digest):
                    report['status'] = 'skipped_duplicate'
                    return self._outbox(report)
                replacing = self._is_replacement(store, doc_id, digest)
                if replacing and not replace:
                    report['errors'].append('A different file with this document ID exists. Enable replace to update it.')
                    return report
                if replacing and report['errors']:
                    return self._outbox(report)
                if replacing and not any(note.is_informative for note in notes):
                    report['errors'].append('Replacement produced no informative notes. The existing document was kept.')
                    return self._outbox(report)
                before = store.snapshot()
                old_ids = list(store.documents.get(doc_id, {}).get('vector_delete_note_ids', []))
                if replacing:
                    old_ids.extend(nid for nid, n in store.notes.items() if n['doc_id'] == doc_id)
                    store.delete_document(doc_id)
                try:
                    for note in notes:
                        try:
                            result = self.graph_agent.ingest_note(note, defer_commit=True)
                            report['notes_informative'] += int(note.is_informative)
                            report['notes_skipped'] += int(result['skipped'])
                            for key in ('nodes_created', 'nodes_merged', 'edges_created'):
                                report[key] += len(result[key])
                        except Exception as exc:
                            report['errors'].append(f'Chunk {note.chunk_index}: {exc}')
                            if replacing:
                                raise
                    report['status'] = 'failed' if report['errors'] else ('replaced' if replacing else 'ingested')
                    store.documents[doc_id] = {'content_hash': digest,
                                               'status': 'partial' if report['errors'] else 'complete'}
                    if old_ids and self.graph_agent.vector_store is not None:
                        store.documents[doc_id]['vector_delete_note_ids'] = list(dict.fromkeys(old_ids))
                    store.commit()
                except Exception:
                    store.restore(before)
                    for key in ('nodes_created', 'nodes_merged', 'edges_created'):
                        report[key] = 0
                    raise
                return self._outbox(report)
        except Exception as exc:
            report['status'] = 'failed'
            report['errors'].append(str(exc))
            return report

    @staticmethod
    def _is_replacement(store, doc_id, digest):
        return (any(n['doc_id'] == doc_id and n['content_hash'] != digest for n in store.notes.values())
                or bool(store.documents.get(doc_id) and store.documents[doc_id]['content_hash'] != digest))

    def _outbox(self, report):
        report['index_errors'].extend(self.graph_agent.flush_vector_outbox())
        report['vector_pending'] = sum(bool(n.get('vector_pending'))
                                      for n in self.graph_agent.graph_store.notes.values())
        report['vector_cleanup_pending'] = sum(bool(d.get('vector_delete_note_ids'))
                                              for d in self.graph_agent.graph_store.documents.values()) + sum(
                                                  bool(n.get('vector_delete_pending'))
                                                  for n in self.graph_agent.graph_store.notes.values())
        return report


def ingest_document(file_path, replace=False, *, pipeline=None):
    if pipeline is None:
        from src.runtime import create_runtime
        with create_runtime() as runtime:
            return runtime.ingestion.ingest_document(file_path, replace)
    return pipeline.ingest_document(file_path, replace)
