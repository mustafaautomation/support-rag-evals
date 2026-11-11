"""Paragraph packer with a character window and overlap."""

import re
from collections.abc import Sequence

from support_rag.models import Chunk, Document


def chunk_text(text: str, size: int = 1800, overlap: int = 200) -> list[str]:
    """Split text into windows of at most `size` characters."""
    if size < 1:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        stripped = text.strip()
        return [stripped] if stripped else []

    packed: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if not current:
            current = paragraph
            continue
        joined = f"{current}\n\n{paragraph}"
        if len(joined) <= size:
            current = joined
            continue
        packed.append(current)
        tail = current[-overlap:] if overlap else ""
        current = f"{tail}\n\n{paragraph}" if tail else paragraph
    if current:
        packed.append(current)

    windows: list[str] = []
    for piece in packed:
        if len(piece) <= size:
            windows.append(piece)
            continue
        start = 0
        while start < len(piece):
            windows.append(piece[start : start + size])
            if start + size >= len(piece):
                break
            start += size - overlap
    return windows


def chunk_documents(
    documents: Sequence[Document],
    size: int = 1800,
    overlap: int = 200,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        parts = chunk_text(document.body, size=size, overlap=overlap)
        for index, part in enumerate(parts):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}:{index}",
                    doc_id=document.doc_id,
                    title=document.title,
                    text=part,
                )
            )
    return chunks
