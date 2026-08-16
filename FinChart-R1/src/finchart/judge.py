"""OpenAI-compatible vision examiner with robust response handling."""

from __future__ import annotations

import base64
import io
import json
import re
import time
from typing import Any

import requests
from PIL import Image

from .config import Phase1Config

VALID_VERDICTS = {"CORRECT", "INCORRECT", "AMBIGUOUS", "POSSIBLE_LABEL_ERROR"}
VALID_ERROR_TYPES = {"NONE", "FORMAT_DIFFERENCE", "VISUAL_EXTRACTION", "NUMERICAL_REASONING", "LOGICAL_REASONING", "COUNTING", "OTHER"}
JUDGE_COLUMNS = {"judge_status": None, "judge_attempts": 0, "judge_last_error": None, "judge_verdict": None, "judge_error_type": None, "judge_confidence": None, "judge_independent_answer": None, "judge_reason": None, "judge_raw_response": None}

JUDGE_SYSTEM_PROMPT = """You are a strict multimodal examiner for a chart question-answering benchmark.
Inspect the chart, solve the question independently, compare the dataset answer and anonymous candidate answer, then decide whether the candidate is actually correct.

Harmless formatting or equivalent numeric formatting is not an error. Do not blindly trust a label if the chart clearly conflicts with it. Do not equate 0.72 and 72 unless chart context establishes percentage conversion. POSSIBLE_LABEL_ERROR is for a likely conflicting or malformed label; AMBIGUOUS is only when the chart/question prevents a confident decision.

Return ONLY valid JSON with exactly: verdict (CORRECT | INCORRECT | AMBIGUOUS | POSSIBLE_LABEL_ERROR), error_type (NONE | FORMAT_DIFFERENCE | VISUAL_EXTRACTION | NUMERICAL_REASONING | LOGICAL_REASONING | COUNTING | OTHER), confidence (0.0-1.0), independent_answer, reason."""


def image_to_data_url(image: Any) -> str:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_chat_completions_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    return url if url.endswith("/chat/completions") else f"{url}/chat/completions"


def extract_message_content(response_json: dict[str, Any]) -> str:
    content = response_json["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(block.get("text", "")) if isinstance(block, dict) else str(block) for block in content)
    return str(content)


def safe_json_from_text(text: str) -> dict[str, Any]:
    """Parse plain, fenced, or embedded JSON without accepting non-object payloads."""
    if text is None:
        raise ValueError("Empty judge content.")
    text = str(text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character == "{":
            try:
                parsed, _ = decoder.raw_decode(text[index:])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse judge JSON. Preview: {text[:800]}")


def normalize_judge_object(parsed: dict[str, Any]) -> dict[str, Any]:
    verdict = str(parsed.get("verdict", "")).upper().strip()
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"Invalid judge verdict: {verdict!r}")
    error_type = str(parsed.get("error_type", "OTHER")).upper().strip()
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return {"judge_verdict": verdict, "judge_error_type": error_type if error_type in VALID_ERROR_TYPES else "OTHER", "judge_confidence": confidence, "judge_independent_answer": str(parsed.get("independent_answer", "")).strip(), "judge_reason": str(parsed.get("reason", "")).strip()}


def call_judge_once(image: Any, question: str, ground_truth: str, candidate_answer: str, config: Phase1Config) -> dict[str, Any]:
    if not config.judge_enabled:
        raise RuntimeError("Set JUDGE_API_KEY, JUDGE_BASE_URL, and JUDGE_MODEL.")
    payload = {"model": config.judge_model, "temperature": 0, "messages": [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": f"Question: {question}\nDataset answer: {ground_truth}\nCandidate answer: {candidate_answer}"},
            {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
        ]},
    ]}
    response = requests.post(build_chat_completions_url(config.judge_base_url), headers={"Authorization": f"Bearer {config.judge_api_key}", "Content-Type": "application/json"}, json=payload, timeout=config.judge_timeout_seconds)
    response.raise_for_status()
    raw = extract_message_content(response.json())
    result = normalize_judge_object(safe_json_from_text(raw))
    result.update({"judge_status": "SUCCESS", "judge_attempts": 1, "judge_last_error": None, "judge_raw_response": raw})
    return result


def call_judge_with_retry(image: Any, question: str, ground_truth: str, candidate_answer: str, config: Phase1Config) -> dict[str, Any]:
    last_error = "Unknown judge error"
    for attempt in range(1, config.judge_max_retries + 1):
        try:
            result = call_judge_once(image, question, ground_truth, candidate_answer, config)
            result["judge_attempts"] = attempt
            return result
        except Exception as error:  # API, transport, schema, and parsing failures are technical failures.
            last_error = f"{type(error).__name__}: {error}"
            if attempt < config.judge_max_retries:
                time.sleep(min(config.judge_backoff_base_seconds * (2 ** (attempt - 1)), config.judge_backoff_max_seconds))
    return {"judge_status": "JUDGE_ERROR", "judge_attempts": config.judge_max_retries, "judge_last_error": last_error, "judge_verdict": None, "judge_error_type": None, "judge_confidence": None, "judge_independent_answer": None, "judge_reason": None, "judge_raw_response": None}
