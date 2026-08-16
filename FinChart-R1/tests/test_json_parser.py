import pytest

from finchart.judge import safe_json_from_text


def test_json_parser_variants() -> None:
    assert safe_json_from_text('{"verdict": "CORRECT"}')["verdict"] == "CORRECT"
    assert safe_json_from_text('```json\n{"verdict": "INCORRECT"}\n```')["verdict"] == "INCORRECT"
    assert safe_json_from_text('Examiner output: {"verdict": "AMBIGUOUS"} done.')["verdict"] == "AMBIGUOUS"


def test_json_parser_rejects_invalid_content() -> None:
    with pytest.raises(ValueError):
        safe_json_from_text("not JSON")
