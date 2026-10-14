"""Build one shared set of stores and per-agent clients."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from threading import RLock

from src import llm_client
from src.ingestion.extraction_agent import ExtractionAgent
from src.ingestion.graph_agent import GraphAgent, MemoryStore
from src.ingestion.pipeline import IngestionPipeline
from src.pipeline import QueryPipeline
from src.stores.errors import StoreError
from src.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
_DEMO_STORES = {}
_DEMO_LOCK = RLock()


@dataclass
class Runtime:
    graph_store: object
    vector_store: object
    ingestion: IngestionPipeline
    query: QueryPipeline
    demo: bool = False
    clients: dict = field(default_factory=dict)

    def close(self):
        try:
            self.graph_store.close()
        finally:
            for client in self.clients.values():
                close = getattr(client, 'close', None)
                if callable(close):
                    close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def create_runtime(*, demo=False, data_path=None):
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    settings = Settings.from_env()
    from src.ingestion.parser import config
    config.ENABLE_OCR = settings.enable_ocr
    clients = {}
    if demo:
        from src.demo import DemoLLM, LocalSearchStore
        base = Path(data_path) if data_path else ROOT / 'data' / 'demo'
        with _DEMO_LOCK:
            key = str(base.resolve())
            if key not in _DEMO_STORES:
                _DEMO_STORES[key] = (MemoryStore(base / 'memory_store.json'),
                                    LocalSearchStore(base / 'search.json'))
            graph, vector = _DEMO_STORES[key]
        clients = {a: DemoLLM(a) for a in ('query', 'reasoning', 'critic', 'extraction', 'graph')}
    else:
        if os.getenv('LLM_FAKE', '0').lower() in ('1', 'true', 'yes', 'on'):
            raise StoreError('LLM_FAKE is for scripted tests. Use the separate demo mode for the app.')
        provider = (os.getenv('LLM_PROVIDER') or 'openai').strip().lower()
        for a in ('query', 'reasoning', 'critic', 'extraction', 'graph'):
            selected = (os.getenv(f'{a.upper()}_PROVIDER') or provider).strip().lower()
            key = {'openai': 'OPENAI_API_KEY', 'anthropic': 'ANTHROPIC_API_KEY'}.get(selected)
            if key and not os.getenv(key):
                raise StoreError(f'Set {key} in .env for the {a} agent')
        from src.stores.neo4j_store import Neo4jStore
        from src.stores.vector_store import VectorStore
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        clients = {a: llm_client.for_agent(a) for a in ('query', 'reasoning', 'critic', 'extraction', 'graph')}
        graph = Neo4jStore()
    try:
        if not demo:
            graph.connect()
            chroma_path = Path(os.getenv('CHROMA_PATH') or 'data/chroma')
            if not chroma_path.is_absolute():
                chroma_path = ROOT / chroma_path
            vector = VectorStore(persist_path=str(chroma_path),
                                 collection_name=os.getenv('CHROMA_COLLECTION') or 'memory_graph',
                                 embedding_function=DefaultEmbeddingFunction())
        writer = GraphAgent(graph, vector, llm=clients['graph'])
        return Runtime(graph, vector, IngestionPipeline(ExtractionAgent(clients['extraction']), writer),
                       QueryPipeline(graph, vector, query_llm=clients['query'],
                                     reasoning_llm=clients['reasoning'], critic_llm=clients['critic'],
                                     top_k=settings.top_k, graph_depth=settings.graph_depth,
                                     min_score=settings.min_score, max_graph_facts=settings.max_graph_facts,
                                     max_retries=settings.max_retries), demo, clients)
    except Exception:
        graph.close()
        for client in clients.values():
            close = getattr(client, 'close', None)
            if callable(close):
                close()
        raise
