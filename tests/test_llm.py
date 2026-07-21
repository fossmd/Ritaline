from ritaline.llm import parse_json_object


def test_parse_fenced_json() -> None:
    payload = parse_json_object('```json\n{"question":"Q?","answer":"A."}\n```')
    assert payload == {"question": "Q?", "answer": "A."}


def test_parse_json_surrounded_by_text() -> None:
    payload = parse_json_object('Result: {"question":"Q?","answer":"A."} done')
    assert payload["answer"] == "A."
