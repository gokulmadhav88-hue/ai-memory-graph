"""Use real persistent Chroma with the existing deterministic test embedder."""
from src.stores.vector_store import VectorStore
from test_vector_store import FakeEmbedder


def test_persistent_chroma_reopens_and_deletes_only_one_document(tmp_path):
    path = str(tmp_path / 'chroma')
    store = VectorStore('integration', persist_path=path, embedding_function=FakeEmbedder())
    store.add('first', 'Atlas is managed by Maya.', {'doc_id': 'atlas'})
    store.add('second', 'Nova is managed by Arun.', {'doc_id': 'nova'})
    reopened = VectorStore('integration', persist_path=path, embedding_function=FakeEmbedder())
    assert {r['note_id'] for r in reopened.query('Atlas', top_k=10)} == {'first', 'second'}
    assert reopened.delete_document('atlas') == 1
    assert [r['note_id'] for r in store.query('Nova')] == ['second']
