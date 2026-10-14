"""Evaluation runner for the memory graph pipeline."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
from time import perf_counter
from statistics import median
import math


def load_cases(path):
    if path.suffix == '.jsonl':
        cases = []
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            case = json.loads(line)
            if not isinstance(case, dict) or not isinstance(case.get('question'), str) or not case['question'].strip():
                raise ValueError(f'Invalid evaluation question on line {number}')
            for key in ('expected_terms', 'expected_doc_ids'):
                if key in case and (not isinstance(case[key], list) or
                                    not all(isinstance(x, str) and x for x in case[key])):
                    raise ValueError(f'{key} must contain strings on line {number}')
            if 'expect_insufficient' in case and type(case['expect_insufficient']) is not bool:
                raise ValueError(f'expect_insufficient must be boolean on line {number}')
            cases.append(case)
        return cases
    cases = []
    for line in path.read_text(encoding='utf-8').splitlines():
        numbered = re.match(r'^\s*\d+[.)]\s+(.+)', line)
        if numbered:
            cases.append({'question': numbered.group(1).strip()})
        elif path.suffix != '.md' and line.strip() and not line.lstrip().startswith('#'):
            cases.append({'question': line.strip()})
    return cases


def check_case(case, response):
    checks = {}
    if 'expected_terms' in case:
        checks['answer_terms'] = all(term.casefold() in response['answer_text'].casefold()
                                     for term in case['expected_terms'])
    if 'expected_doc_ids' in case:
        found = {source['doc_id'] for source in response['sources']}
        checks['source_documents'] = set(case['expected_doc_ids']) <= found
    if 'expect_insufficient' in case:
        checks['abstention'] = response['insufficient_evidence'] == case['expect_insufficient']
    return checks


def run_eval(eval_file: str | Path, *, pipeline=None) -> dict:
    path = Path(eval_file)
    if not path.exists():
        raise FileNotFoundError(path)
    cases = load_cases(path)
    if pipeline is None:
        from src.runtime import create_runtime
        with create_runtime() as runtime:
            return run_eval(path, pipeline=runtime.query)
    results = []
    for case in cases:
        question = case['question']
        started = perf_counter()
        try:
            response = pipeline.answer_question(question)
            results.append({'question': question, 'response': response, 'error': None,
                            'latency_seconds': perf_counter() - started,
                            'checks': check_case(case, response)})
        except Exception as exc:
            results.append({'question': question, 'response': None, 'error': str(exc),
                            'latency_seconds': perf_counter() - started, 'checks': {}})
    latencies = sorted(r['latency_seconds'] for r in results)
    return {
        "eval_file": str(path),
        "total_questions": len(cases),
        "results": results,
        "answered": sum(bool(r['response'] and not r['response']['insufficient_evidence']) for r in results),
        "errors": sum(r['error'] is not None for r in results),
        "annotated_cases": sum(any(key in case for key in
                                   ('expected_terms', 'expected_doc_ids', 'expect_insufficient')) for case in cases),
        "passed_annotated_cases": sum(r['error'] is None and bool(r['checks']) and all(r['checks'].values())
                                      for r in results),
        "median_latency_seconds": median(latencies) if latencies else None,
        "p95_latency_seconds": latencies[max(0, math.ceil(len(latencies) * 0.95) - 1)] if latencies else None,
        "evaluation_note": "Term, source, and abstention checks are explicit heuristics, not semantic answer accuracy.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?', default='docs/eval_questions.md')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    from src.runtime import create_runtime
    with create_runtime(demo=args.demo) as runtime:
        print(json.dumps(run_eval(args.file, pipeline=runtime.query), ensure_ascii=False, indent=2))
