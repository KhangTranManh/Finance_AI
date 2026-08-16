"""Conservative deterministic answer matching."""

from __future__ import annotations

import re
import string


def normalize_answer(text: object) -> str:
    text = "" if text is None else str(text)
    text = text.lower().strip()
    text = re.sub(r"^\s*final\s+answer\s*:\s*", "", text)
    text = re.sub(r"^\s*answer\s*:\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*/\s*", "/", text)
    return text.strip(string.whitespace + ".,;:!?")


def try_parse_number(text: object) -> float | None:
    try:
        return float(normalize_answer(text).replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return None


def deterministic_is_correct(
    prediction: object, ground_truth: object, tolerance: float = 1e-6
) -> bool:
    """Match only safe formatting and numeric equivalents without semantic guesses."""
    prediction_normalized = normalize_answer(prediction)
    truth_normalized = normalize_answer(ground_truth)
    if prediction_normalized == truth_normalized:
        return True
    prediction_number = try_parse_number(prediction_normalized)
    truth_number = try_parse_number(truth_normalized)
    return (
        prediction_number is not None
        and truth_number is not None
        and abs(prediction_number - truth_number) <= tolerance
    )
