from finchart.evaluator import deterministic_is_correct


def test_safe_equivalences() -> None:
    assert deterministic_is_correct("Blue.", "blue")
    assert deterministic_is_correct("25.0", "25")
    assert deterministic_is_correct("Not too much/ not at all", "Not too much/not at all")


def test_contextual_percentage_is_not_a_deterministic_match() -> None:
    assert not deterministic_is_correct("0.72", "72")
