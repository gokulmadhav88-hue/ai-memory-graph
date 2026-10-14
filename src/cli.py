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
    sub.add_parser('check')
    args = parser.parse_args()
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
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get('status') == 'failed':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
