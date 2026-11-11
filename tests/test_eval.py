import asyncio
import json
from pathlib import Path

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NonLLMContextRecall
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    StringPresence,
)

from support_rag.eval.cli import main
from support_rag.eval.dataset import attach_contexts, load_questions
from support_rag.eval.gate import apply_gate, load_thresholds
from support_rag.eval.live import LIVE_METRIC_NAMES, score_live
from support_rag.eval.metrics import score_answers
from support_rag.eval.report import render_markdown
from support_rag.eval.snapshots import dump_answers, load_answers
from support_rag.models import Answer, Chunk, Retrieved
from support_rag.paths import repo_root
from support_rag.pipeline import build_pipeline, load_chunks


def _prepared() -> tuple[list, list]:
    chunks = load_chunks()
    questions = attach_contexts(load_questions(), chunks)
    return questions, chunks


def test_dataset_size_and_stored_contexts() -> None:
    questions, chunks = _prepared()
    documents = {chunk.doc_id for chunk in chunks}
    assert 15 <= len(documents) <= 30
    assert 30 <= len(questions) <= 50
    raw_lines = (
        (repo_root() / "data" / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    )
    raw = [json.loads(line) for line in raw_lines if line.strip()]
    behaviors = {question.expected_behavior for question in questions}
    assert {"answer", "refuse", "escalate", "unknown"} <= behaviors
    for row, question in zip(raw, questions, strict=True):
        assert row["reference_contexts"] == list(question.reference_contexts)
        if question.reference_doc_ids:
            assert question.reference_contexts
            assert question.presence_phrase in question.reference_contexts[0]
        else:
            assert question.reference_contexts == ()


def test_ragas_non_llm_metrics_score_exact_strings() -> None:
    presence = StringPresence().score(reference="14 days", response="Refunded within 14 days.")
    assert presence.value == 1.0
    metric = NonLLMContextRecall()
    matched = asyncio.run(
        metric.single_turn_ascore(
            SingleTurnSample(
                retrieved_contexts=["Annual plans can be refunded within 14 days."],
                reference_contexts=["Annual plans can be refunded within 14 days."],
            )
        )
    )
    missed = asyncio.run(
        metric.single_turn_ascore(
            SingleTurnSample(
                retrieved_contexts=["Check the phone notification permission."],
                reference_contexts=["Annual plans can be refunded within 14 days."],
            )
        )
    )
    assert matched == 1.0
    assert missed == 0.0


def test_collections_api_exports_the_judge_metrics() -> None:
    assert LIVE_METRIC_NAMES == (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )
    assert Faithfulness.__name__ == "Faithfulness"
    assert AnswerRelevancy.__name__ == "AnswerRelevancy"
    assert ContextPrecision.__name__ == "ContextPrecision"
    assert ContextRecall.__name__ == "ContextRecall"


def test_live_context_metrics_skip_questions_without_a_source_doc() -> None:
    class _Result:
        def __init__(self, value: float) -> None:
            self.value = value

    class _Scorer:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def ascore(self, **kwargs: object) -> _Result:
            self.calls.append(str(kwargs["user_input"]))
            return _Result(1.0)

    questions, _chunks = _prepared()
    sample = questions[0]
    unknown = next(question for question in questions if question.expected_behavior == "unknown")
    from support_rag.eval.dataset import EvalQuestion

    labelled = [
        EvalQuestion(
            id="a",
            topic="billing",
            question=sample.question,
            reference=sample.reference,
            reference_doc_ids=("plans",),
            reference_contexts=("Plans\n\nbody",),
            expected_behavior="answer",
            presence_phrase="$12",
        ),
        EvalQuestion(
            id="b",
            topic="out-of-scope",
            question=unknown.question,
            reference=unknown.reference,
            reference_doc_ids=(),
            reference_contexts=(),
            expected_behavior="unknown",
            presence_phrase="I don't know",
        ),
    ]
    answers = [
        Answer(
            item.question,
            "reply",
            (Retrieved(Chunk("plans:0", "plans", "Plans", "body"), 1.0),),
            None,
            "x",
        )
        for item in labelled
    ]
    faith, relevancy, precision, recall = _Scorer(), _Scorer(), _Scorer(), _Scorer()
    scores = asyncio.run(
        score_live(
            labelled,
            answers,
            faithfulness=faith,
            relevancy=relevancy,
            precision=precision,
            recall=recall,
        )
    )
    assert scores == {
        "faithfulness": 1.0,
        "answer_relevancy": 1.0,
        "context_precision": 1.0,
        "context_recall": 1.0,
    }
    assert faith.calls == [sample.question, unknown.question]
    assert precision.calls == [sample.question]
    assert recall.calls == [sample.question]


def test_improved_passes_gate_and_broken_variants_fail() -> None:
    questions, chunks = _prepared()
    thresholds = load_thresholds()["offline"]
    reports = {}
    for variant in ("improved", "bad-retriever", "no-guardrail"):
        pipeline = build_pipeline(variant, chunks=chunks)
        answers = [pipeline.answer(question.question) for question in questions]
        reports[variant] = score_answers(questions, answers)
    improved = apply_gate(reports["improved"].scores, thresholds)
    bad = apply_gate(reports["bad-retriever"].scores, thresholds)
    unguarded = apply_gate(reports["no-guardrail"].scores, thresholds)
    assert improved.passed
    assert reports["improved"].failures == ()
    assert not bad.passed
    assert not unguarded.passed
    assert any(item.startswith("retrieval_hit_rate") for item in bad.failures)
    assert any(item.startswith("refusal_correctness") for item in unguarded.failures)


def test_snapshot_matches_the_pipeline(tmp_path: Path) -> None:
    questions, chunks = _prepared()
    pipeline = build_pipeline("improved", chunks=chunks)
    answers = [pipeline.answer(question.question) for question in questions]
    path = tmp_path / "improved.jsonl"
    dump_answers(path, questions, answers)
    loaded = load_answers(path)
    assert [item.response for item in loaded] == [item.response for item in answers]
    assert [item.chunk.doc_id for item in loaded[0].retrieved] == [
        item.chunk.doc_id for item in answers[0].retrieved
    ]
    committed = load_answers(repo_root() / "data" / "snapshots" / "improved.jsonl")
    assert [item.response for item in committed] == [item.response for item in answers]
    assert [[hit.chunk.doc_id for hit in item.retrieved] for item in committed] == [
        [hit.chunk.doc_id for hit in item.retrieved] for item in answers
    ]


def test_cli_scores_the_recorded_improved_run_and_rejects_the_bad_one() -> None:
    root = repo_root()
    assert (
        main(["--mode", "offline", "--from-snapshot", str(root / "data/snapshots/improved.jsonl")])
        == 0
    )
    assert (
        main(
            [
                "--mode",
                "offline",
                "--from-snapshot",
                str(root / "data/snapshots/bad-retriever.jsonl"),
            ]
        )
        == 1
    )
    assert main(["--mode", "live"]) == 2


def test_report_mentions_the_measured_improved_score() -> None:
    questions, chunks = _prepared()
    pipeline = build_pipeline("improved", chunks=chunks)
    report = score_answers(
        questions, [pipeline.answer(question.question) for question in questions]
    )
    text = (repo_root() / "docs" / "eval-report.md").read_text(encoding="utf-8")
    assert f"{report.scores['mrr']:.3f}" in text
    assert "bad-retriever" in text
    assert "No LLM judge scores" in text
    rendered = render_markdown(
        generated_on="2026-11-11",
        variants={"improved": report, "bad-retriever": report, "no-guardrail": report},
        gate=apply_gate(report.scores, load_thresholds()["offline"]),
        note="note",
    )
    assert "String presence" in rendered
