"""Headless ingestion, questions, and service checks."""
import argparse
import json

from src.runtime import create_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true')
    sub = parser.add_subparsers(dest='command', required=True)
    ingest = sub.add_parser('ingest')
    ingest.add_argument('path')
    ingest.add_argument('--replace', action='store_true')
    ask = sub.add_parser('ask')
    ask.add_argument('question')
    check = sub.add_parser('check')
    check.add_argument('--probe-model', action='store_true', help='Makes a small paid request to each distinct model')
    sub.add_parser('warmup')
    args = parser.parse_args()
    if args.command == 'check' and args.demo and args.probe_model:
        parser.error('--probe-model checks real models; omit --demo')
    if args.command == 'warmup':
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        vectors = DefaultEmbeddingFunction()(['Embedding readiness check'])
        print(json.dumps({'embedding_ready': True, 'dimensions': len(vectors[0])}))
        return
    with create_runtime(demo=args.demo) as runtime:
        if args.command == 'ingest':
            result = runtime.ingestion.ingest_document(args.path, args.replace)
        elif args.command == 'ask':
            result = runtime.query.answer_question(args.question)
        else:
            runtime.graph_store.refresh()
            if not args.demo:
                runtime.vector_store._get_collection()
            result = {'connected': True, 'demo': args.demo,
                      'documents': len(runtime.graph_store.documents)}
            if args.probe_model:
                seen = set()
                for client in runtime.clients.values():
                    identity = (getattr(client, 'provider', None), getattr(client, 'model', None))
                    if identity in seen:
                        continue
                    seen.add(identity)
                    raw = client.complete_json('Return JSON {"ready":true}', max_tokens=50)
                    if raw.get('ready') is not True and not args.demo:
                        raise RuntimeError('Model probe did not return the expected readiness response')
                result['models_checked'] = len(seen)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get('status') == 'failed':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
