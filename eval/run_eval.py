"""Evaluation runner for the memory graph pipeline."""

from __future__ import annotations

from pathlib import Path


def run_eval(eval_file: str | Path) -> dict:
    path = Path(eval_file)
    questions = []
    if path.exists():
        questions = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    return {
        "eval_file": str(path),
        "questions": questions,
        "total_questions": len(questions),
        "score": 0,
    }


if __name__ == "__main__":
    result = run_eval("docs/eval_questions.md")
    print(result)
