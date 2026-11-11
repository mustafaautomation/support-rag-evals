"""LLM-judged RAGAS metrics. This module is used only in live mode."""

import asyncio
import os
from collections.abc import Sequence

from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from support_rag.eval.dataset import EvalQuestion
from support_rag.models import Answer

LIVE_METRIC_NAMES: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


async def score_live(
    questions: Sequence[EvalQuestion],
    answers: Sequence[Answer],
    *,
    faithfulness: object,
    relevancy: object,
    precision: object,
    recall: object,
) -> dict[str, float]:
    """Average the four collections-API RAGAS metrics.

    Context precision and context recall are computed only for questions that
    name a reference document. Out-of-scope questions have no source document
    to recall. Faithfulness and answer relevancy still cover every question.
    """
    by_question = {item.question: item for item in answers}
    faith_scores: list[float] = []
    relevancy_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    for question in questions:
        answer = by_question[question.question]
        contexts = [item.chunk.embedded_text for item in answer.retrieved]
        faith = await faithfulness.ascore(  # type: ignore[attr-defined]
            user_input=question.question,
            response=answer.response,
            retrieved_contexts=contexts,
        )
        relevant = await relevancy.ascore(  # type: ignore[attr-defined]
            user_input=question.question,
            response=answer.response,
        )
        faith_scores.append(float(faith.value))
        relevancy_scores.append(float(relevant.value))
        if not question.reference_doc_ids:
            continue
        precise = await precision.ascore(  # type: ignore[attr-defined]
            user_input=question.question,
            reference=question.reference,
            retrieved_contexts=contexts,
        )
        recalled = await recall.ascore(  # type: ignore[attr-defined]
            user_input=question.question,
            retrieved_contexts=contexts,
            reference=question.reference,
        )
        precision_scores.append(float(precise.value))
        recall_scores.append(float(recalled.value))
    return {
        "faithfulness": _mean(faith_scores),
        "answer_relevancy": _mean(relevancy_scores),
        "context_precision": _mean(precision_scores),
        "context_recall": _mean(recall_scores),
    }


def run_live_judge(
    questions: Sequence[EvalQuestion],
    answers: Sequence[Answer],
) -> dict[str, float]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set, so the live judge cannot run.")
    from openai import AsyncOpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory

    client = AsyncOpenAI()
    model = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-mini")
    embed_model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    llm = llm_factory(model, client=client)
    embeddings = embedding_factory("openai", model=embed_model, client=client)
    return asyncio.run(
        score_live(
            questions,
            answers,
            faithfulness=Faithfulness(llm=llm),
            relevancy=AnswerRelevancy(
                llm=llm,
                embeddings=embeddings,  # type: ignore[arg-type]
            ),
            precision=ContextPrecision(llm=llm),
            recall=ContextRecall(llm=llm),
        )
    )


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Live metric received no rows")
    return sum(values) / len(values)
