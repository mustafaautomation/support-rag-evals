from support_rag.chunking import chunk_documents, chunk_text
from support_rag.models import Document


def test_short_text_is_one_chunk() -> None:
    assert chunk_text("One paragraph about refunds.") == ["One paragraph about refunds."]


def test_empty_text_has_no_chunks() -> None:
    assert chunk_text("   ") == []


def test_overlap_window_keeps_the_tail() -> None:
    text = "A" * 50 + "\n\n" + "B" * 50
    chunks = chunk_text(text, size=60, overlap=10)
    assert len(chunks) >= 2
    assert chunks[1].startswith("A" * 10)


def test_documents_keep_ids() -> None:
    document = Document("plans", "Plans", "Grove costs twelve dollars.")
    chunks = chunk_documents([document], size=1800, overlap=200)
    assert chunks[0].chunk_id == "plans:0"
    assert chunks[0].embedded_text.startswith("Plans\n\n")
