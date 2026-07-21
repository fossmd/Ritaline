from pathlib import Path

from ritaline.documents import load_document


def test_load_text_document(tmp_path: Path) -> None:
    path = tmp_path / "source.txt"
    path.write_text("Hello\n\n\nworld", encoding="utf-8")
    document = load_document(path)
    assert document.pages[0].text == "Hello\n\nworld"
