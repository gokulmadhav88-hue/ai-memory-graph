"""Build one shared set of stores and per-agent clients."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from src import llm_client
from src.ingestion.extraction_agent import ExtractionAgent
from src.ingestion.graph_agent import GraphAgent, MemoryStore
from src.ingestion.pipeline import IngestionPipeline
from src.pipeline import QueryPipeline
from src.stores.errors import StoreError

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Runtime:
    graph_store: object
    vector_store: object
    ingestion: IngestionPipeline
    query: QueryPipeline
    demo: bool = False

    def close(self):
        self.graph_store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def create_runtime(*, demo=False, data_path=None):
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    if demo:
        from src.demo import DemoLLM, LocalSearchStore
        base = Path(data_path) if data_path else ROOT / 'data' / 'demo'
        graph = MemoryStore(base / 'memory_store.json')
        vector = LocalSearchStore(base / 'search.json')
        clients = {a: DemoLLM(a) for a in ('query', 'reasoning', 'critic', 'extraction', 'graph')}
    else:
        if os.getenv('LLM_FAKE', '0').lower() in ('1', 'true', 'yes', 'on'):
            raise StoreError('LLM_FAKE is for scripted tests. Use the separate demo mode for the app.')
        provider = os.getenv('LLM_PROVIDER', 'openai').lower()
        for a in ('query', 'reasoning', 'critic', 'extraction', 'graph'):
            selected = (os.getenv(f'{a.upper()}_PROVIDER') or provider).lower()
            key = {'openai': 'OPENAI_API_KEY', 'anthropic': 'ANTHROPIC_API_KEY'}.get(selected)
            if key and not os.getenv(key):
                raise StoreError(f'Set {key} in .env for the {a} agent')
        from src.stores.neo4j_store import Neo4jStore
        from src.stores.vector_store import VectorStore
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        clients = {a: llm_client.for_agent(a) for a in ('query', 'reasoning', 'critic', 'extraction', 'graph')}
        graph = Neo4jStore()
        graph.connect()
        vector = VectorStore(persist_path=os.getenv('CHROMA_PATH', str(ROOT / 'data' / 'chroma')),
                             collection_name=os.getenv('CHROMA_COLLECTION', 'memory_graph'),
                             embedding_function=DefaultEmbeddingFunction())
    try:
        writer = GraphAgent(graph, vector, llm=clients['graph'])
        return Runtime(graph, vector, IngestionPipeline(ExtractionAgent(clients['extraction']), writer),
                       QueryPipeline(graph, vector, query_llm=clients['query'],
                                     reasoning_llm=clients['reasoning'], critic_llm=clients['critic']), demo)
    except Exception:
        graph.close()
        raise
