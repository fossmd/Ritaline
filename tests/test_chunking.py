from pathlib import Path

from ritaline.chunking import chunk_document
from ritaline.models import DocumentPage, SourceDocument


def test_chunking_preserves_page_ranges() -> None:
    document = SourceDocument(
        path=Path("manual.pdf"),
        pages=(
            DocumentPage(number=1, text="A sentence. " * 80),
            DocumentPage(number=2, text="B sentence. " * 80),
        ),
    )

    chunks = chunk_document(
        document,
        chunk_size_chars=500,
        chunk_overlap_chars=50,
        min_chunk_chars=50,
    )

    assert len(chunks) > 1
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert all(chunk.text for chunk in chunks)
