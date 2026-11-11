"""Evaluation command: offline gate, snapshot scoring, and optional live judge."""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

from support_rag.eval.dataset import EvalQuestion, attach_contexts, load_questions
from support_rag.eval.gate import GateResult, apply_gate, load_thresholds
from support_rag.eval.live import run_live_judge
from support_rag.eval.metrics import MetricReport, score_answers
from support_rag.eval.report import render_html, render_markdown
from support_rag.eval.snapshots import dump_answers, load_answers
from support_rag.models import Chunk
from support_rag.paths import repo_root
from support_rag.pipeline import VARIANTS, Pipeline, build_pipeline, load_chunks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the Sableboard support RAG pipeline.")
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    parser.add_argument("--variant", choices=VARIANTS, default="improved")
    parser.add_argument("--from-snapshot", type=Path, default=None)
    parser.add_argument("--dump-snapshot", type=Path, default=None)
    parser.add_argument("--write-report", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    root = repo_root()
    chunks = load_chunks(root)
    questions = attach_contexts(load_questions(root / "data" / "eval" / "questions.jsonl"), chunks)
    thresholds = load_thresholds(root / "config" / "thresholds.yaml")

    if args.write_report is not None:
        return _write_comparison(questions, chunks, thresholds["offline"], args.write_report)

    if args.mode == "live" and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Live eval was not run.", file=sys.stderr)
        return 2

    if args.from_snapshot is not None and args.mode == "offline":
        answers = load_answers(args.from_snapshot)
    elif args.mode == "live":
        pipeline = _live_pipeline(args.variant, chunks)
        answers = [pipeline.answer(question.question) for question in questions]
    else:
        pipeline = build_pipeline(args.variant, chunks=chunks)
        answers = [pipeline.answer(question.question) for question in questions]
    if args.dump_snapshot is not None:
        dump_answers(args.dump_snapshot, questions, answers)

    report = score_answers(questions, answers)
    active = dict(thresholds["offline"])
    if args.mode == "live":
        live_scores = run_live_judge(questions, answers)
        report_scores = dict(report.scores)
        report_scores.update(live_scores)
        active.update(thresholds["live"])
    else:
        report_scores = report.scores
    gate = apply_gate(report_scores, active)
    _print_scores(report_scores, gate)
    if args.verbose:
        for item in report.failures:
            print(f"{item.id}: {'; '.join(item.failures)}")
    return 0 if gate.passed else 1


def _live_pipeline(variant: str, chunks: list[Chunk]) -> Pipeline:
    from openai import OpenAI

    from support_rag.embeddings import OpenAIEmbedder
    from support_rag.providers import OpenAIGenerator

    client = OpenAI()
    chat_model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    embed_model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    return build_pipeline(
        variant,
        chunks=chunks,
        generator=OpenAIGenerator(client, model=chat_model),
        embedder=OpenAIEmbedder(client, model=embed_model),
    )


def _write_comparison(
    questions: list[EvalQuestion],
    chunks: list[Chunk],
    thresholds: dict[str, float],
    directory: Path,
) -> int:
    variants: dict[str, MetricReport] = {}
    for name in ("bad-retriever", "no-guardrail", "improved"):
        pipeline = build_pipeline(name, chunks=chunks)
        answers = [pipeline.answer(question.question) for question in questions]
        variants[name] = score_answers(questions, answers)
        dump_answers(repo_root() / "data" / "snapshots" / f"{name}.jsonl", questions, answers)
    gate = apply_gate(variants["improved"].scores, thresholds)
    note = (
        "These numbers come from one offline run of the deterministic pipeline "
        "on the committed synthetic help center. They are not LLM-judged RAGAS scores."
    )
    generated_on = date.today().isoformat()
    markdown = render_markdown(
        generated_on=generated_on,
        variants=variants,
        gate=gate,
        note=note,
    )
    html = render_html(generated_on=generated_on, variants=variants, gate=gate, note=note)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "eval-report.md").write_text(markdown, encoding="utf-8")
    (directory / "eval-report.html").write_text(html, encoding="utf-8")
    _print_scores(variants["improved"].scores, gate)
    return 0 if gate.passed else 1


def _print_scores(scores: dict[str, float], gate: GateResult) -> None:
    for name in sorted(scores):
        limit = gate.thresholds.get(name)
        limit_text = f" (threshold {limit:.3f})" if limit is not None else ""
        print(f"{name}: {scores[name]:.3f}{limit_text}")
    print("gate: pass" if gate.passed else "gate: fail")
    for failure in gate.failures:
        print(f"  {failure}")
