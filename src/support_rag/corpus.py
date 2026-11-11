"""Load the synthetic help-center markdown files."""

from pathlib import Path

import yaml

from support_rag.models import Document


def load_corpus(directory: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(directory.glob("*.md")):
        documents.append(_read_document(path))
    if not documents:
        raise FileNotFoundError(f"No markdown documents in {directory}")
    ids = [document.doc_id for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Corpus document ids must be unique")
    return documents


def _read_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        raise ValueError(f"{path.name} is missing front matter")
    _, front_matter, body = raw.split("---", 2)
    loaded = yaml.safe_load(front_matter)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.name} front matter is not a mapping")
    doc_id = loaded.get("id")
    title = loaded.get("title")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError(f"{path.name} needs a string id")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"{path.name} needs a string title")
    text = body.strip()
    if not text:
        raise ValueError(f"{path.name} has an empty body")
    return Document(doc_id=doc_id.strip(), title=title.strip(), body=text)
