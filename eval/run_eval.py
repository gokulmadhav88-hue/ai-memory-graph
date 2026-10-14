"""Evaluation runner for the memory graph pipeline."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re


def run_eval(eval_file: str | Path, *, pipeline=None) -> dict:
    path = Path(eval_file)
    if not path.exists():
        raise FileNotFoundError(path)
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        numbered = re.match(r'^\s*\d+[.)]\s+(.+)', line)
        if numbered:
            questions.append(numbered.group(1).strip())
        elif path.suffix != '.md' and line.strip() and not line.lstrip().startswith('#'):
            questions.append(line.strip())
    if pipeline is None:
        from src.runtime import create_runtime
        with create_runtime() as runtime:
            return run_eval(path, pipeline=runtime.query)
    results = []
    for question in questions:
        try:
            response = pipeline.answer_question(question)
            results.append({'question': question, 'response': response, 'error': None})
        except Exception as exc:
            results.append({'question': question, 'response': None, 'error': str(exc)})
    return {
        "eval_file": str(path),
        "total_questions": len(questions),
        "results": results,
        "answered": sum(bool(r['response'] and not r['response']['insufficient_evidence']) for r in results),
        "errors": sum(r['error'] is not None for r in results),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?', default='docs/eval_questions.md')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    from src.runtime import create_runtime
    with create_runtime(demo=args.demo) as runtime:
        print(json.dumps(run_eval(args.file, pipeline=runtime.query), ensure_ascii=False, indent=2))
