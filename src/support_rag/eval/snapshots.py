"""Read and write JSONL recordings of pipeline outputs."""

import json
from collections.abc import Sequence
from pathlib import Path

from support_rag.eval.dataset import EvalQuestion
from support_rag.models import Answer, Chunk, Retrieved


def dump_answers(path: Path, questions: Sequence[EvalQuestion], answers: Sequence[Answer]) -> None:
    by_question = {item.question: item for item in answers}
    lines: list[str] = []
    for question in questions:
        answer = by_question[question.question]
        payload = {
            "id": question.id,
            "question": question.question,
            "response": answer.response,
            "retrieved_doc_ids": [item.chunk.doc_id for item in answer.retrieved],
            "retrieved_contexts": [item.chunk.embedded_text for item in answer.retrieved],
            "guardrail": answer.guardrail,
            "variant": answer.variant,
        }
        lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_answers(path: Path) -> list[Answer]:
    answers: list[Answer] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        retrieved: list[Retrieved] = []
        doc_ids = payload["retrieved_doc_ids"]
        contexts = payload["retrieved_contexts"]
        for index, (doc_id, context) in enumerate(zip(doc_ids, contexts, strict=True)):
            title, _, text = str(context).partition("\n\n")
            retrieved.append(
                Retrieved(
                    chunk=Chunk(
                        chunk_id=f"{doc_id}:{index}",
                        doc_id=str(doc_id),
                        title=title,
                        text=text,
                    ),
                    score=0.0,
                )
            )
        answers.append(
            Answer(
                question=str(payload["question"]),
                response=str(payload["response"]),
                retrieved=tuple(retrieved),
                guardrail=payload.get("guardrail"),
                variant=str(payload.get("variant", "")),
            )
        )
    return answers
