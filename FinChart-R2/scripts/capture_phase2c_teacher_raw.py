"""Capture raw Phase 2C teacher annotations without filtering them into DPO data.

This train-only collector saves the original teacher message (including separate
reasoning fields exposed by compatible APIs), plus a best-effort parsed JSON
object. It deliberately does not reject partial schemas: later analysis decides
which records are representation resolutions, DPO candidates, or manual review.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "results" / "finchart_r2_phase2c_train_mining"
INPUT_NAME = "dpo_train_error.jsonl"
ALL_NAME = "dpo_train_teacher_raw_capture.jsonl"
REPORT_NAME = "dpo_train_teacher_raw_capture_report.json"

MINIMAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["MODEL_ERROR", "REPRESENTATION_EQUIVALENT", "DATASET_AMBIGUITY", "INCONCLUSIVE"],
        },
        "correct_final_answer": {"type": ["string", "null"]},
        "corrected_response": {"type": ["string", "null"]},
        "bbox": {
            "type": ["array", "null"],
            "items": {"type": "integer", "minimum": 0, "maximum": 1000},
            "minItems": 4,
            "maxItems": 4,
        },
    },
    "required": ["verdict", "correct_final_answer", "corrected_response", "bbox"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DIR / INPUT_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--structured-mode", choices=("json_schema", "json_object"), default="json_object")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def checkpoint(path: Path, rows: dict[int, dict[str, Any]]) -> None:
    write_jsonl(path, [rows[index] for index in sorted(rows)])


def image_as_base64(image: Any) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_json_response(content: Any) -> dict[str, Any]:
    text = str(content or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Teacher content does not contain a parseable JSON object")


def env_value(primary: str, fallback: str = "") -> str:
    return os.getenv(primary, "").strip() or os.getenv(fallback, "").strip()


def teacher_config() -> dict[str, Any]:
    load_dotenv(ROOT / ".env")  # environment variables retain precedence; values are never logged.
    provider = env_value("VISUAL_TEACHER_PROVIDER").lower() or "openai_compatible"
    key = env_value("VISUAL_TEACHER_API_KEY", "TEACHER_API_KEY")
    model = env_value("VISUAL_TEACHER_MODEL", "TEACHER_MODEL")
    base_url = env_value("VISUAL_TEACHER_BASE_URL", "TEACHER_BASE_URL")
    attempts = int(env_value("VISUAL_TEACHER_MAX_ATTEMPTS") or "2")
    if provider != "openai_compatible":
        raise ValueError("Raw-capture notebook currently expects an OpenAI-compatible vision endpoint.")
    if not key or not model or not base_url:
        raise RuntimeError("Teacher API key, base URL, or model is missing.")
    if attempts < 1:
        raise ValueError("VISUAL_TEACHER_MAX_ATTEMPTS must be positive.")
    return {"provider": provider, "key": key, "model": model, "base_url": base_url, "max_attempts": attempts}


def prompt_for(row: dict[str, Any]) -> str:
    return f"""You are an offline chart-QA teacher collecting an annotation for later review.
Inspect the chart independently. Return only one compact JSON object with exactly these keys:
verdict, correct_final_answer, corrected_response, bbox.

verdict must be MODEL_ERROR, REPRESENTATION_EQUIVALENT, DATASET_AMBIGUITY, or INCONCLUSIVE.
For MODEL_ERROR, corrected_response must use four concise lines:
Relevant values: ...
Operation: ...
Calculation: ...
Answer: ...
For every other verdict, set correct_final_answer and corrected_response to null.
bbox is [ymin, xmin, ymax, xmax] normalized 0-1000, or null. Do not add markdown.

Question: {row['question']}
ChartQA reference answer: {row['ground_truth']}
SFT raw response: {row['sft_prediction_raw']}
Extracted SFT answer: {row.get('extracted_final_answer')}"""


def call_teacher(row: dict[str, Any], image: Any, config: dict[str, Any], structured_mode: str) -> dict[str, Any]:
    url = config["base_url"].rstrip("/")
    if not url.endswith("chat/completions"):
        url += "/chat/completions"
    response_format: dict[str, Any]
    if structured_mode == "json_schema":
        response_format = {"type": "json_schema", "json_schema": {"name": "phase2c_raw_capture", "strict": False, "schema": MINIMAL_SCHEMA}}
    else:
        response_format = {"type": "json_object"}
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {config['key']}", "Content-Type": "application/json"},
        json={
            "model": config["model"], "temperature": 0, "max_tokens": 700,
            "response_format": response_format,
            "messages": [
                {"role": "system", "content": "Return only the compact JSON object requested by the user."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_for(row)},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_as_base64(image)}},
                ]},
            ],
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


def capture_one(row: dict[str, Any], image: Any, config: dict[str, Any], structured_mode: str) -> dict[str, Any]:
    raw_message: dict[str, Any] | None = None
    try:
        last_error: Exception | None = None
        for attempt in range(config["max_attempts"]):
            try:
                raw_message = call_teacher(row, image, config, structured_mode)
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 < config["max_attempts"]:
                    time.sleep(2**attempt)
        else:
            raise RuntimeError(f"Teacher request failed: {type(last_error).__name__}")
        content = raw_message.get("content")
        try:
            parsed, status, error = parse_json_response(content), "RAW_CAPTURED", None
        except Exception as exc:
            parsed, status, error = None, "RAW_UNPARSEABLE", f"{type(exc).__name__}: {exc}"[:500]
    except Exception as exc:
        parsed, status, error = None, "TEACHER_ERROR", f"{type(exc).__name__}: {exc}"[:500]
    return {
        **row,
        "capture_protocol": "phase2c_teacher_raw_v1",
        "teacher_model": config["model"],
        "structured_mode": structured_mode,
        "teacher_status": status,
        "teacher_response_raw": raw_message,
        "teacher_content_raw": raw_message.get("content") if raw_message else None,
        "teacher_reasoning_raw": (raw_message.get("reasoning") or raw_message.get("reasoning_content")) if raw_message else None,
        "teacher_json_parsed": parsed,
        "teacher_error": error,
        "dpo_pairs_created": False,
    }


def main() -> None:
    args = parse_args()
    if args.limit < 1 or not 1 <= args.workers <= 4:
        raise ValueError("--limit must be positive and --workers must be 1-4")
    source = load_jsonl(args.input)
    required = {"dataset_index", "image_split", "image_index", "question", "ground_truth", "sft_prediction_raw"}
    for position, row in enumerate(source, 1):
        if missing := required - set(row):
            raise ValueError(f"Input line {position} missing {sorted(missing)}")
        if row["image_split"] != "train":
            raise ValueError("Raw capture is restricted to ChartQA train records")
    selected = source[:args.limit]
    output_dir = args.output_dir.resolve()
    all_path = output_dir / ALL_NAME
    existing = {} if args.no_resume or not all_path.exists() else {int(row["dataset_index"]): row for row in load_jsonl(all_path)}
    pending = [row for row in selected if int(row["dataset_index"]) not in existing or (args.retry_failed and existing[int(row["dataset_index"])].get("teacher_status") in {"TEACHER_ERROR", "RAW_UNPARSEABLE"})]
    config = teacher_config()
    dataset = load_dataset("HuggingFaceM4/ChartQA", split="train")

    def resolve_image(row: dict[str, Any]) -> Any:
        example = dataset[int(row["image_index"])]
        source_question = str(example.get("query", example.get("question", ""))).strip()
        if source_question != str(row["question"]).strip():
            raise ValueError(f"ChartQA question mismatch at dataset_index={row['dataset_index']}")
        return example["image"]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(capture_one, row, resolve_image(row), config, args.structured_mode) for row in pending]
        for count, future in enumerate(as_completed(futures), 1):
            record = future.result()
            existing[int(record["dataset_index"])] = record
            checkpoint(all_path, existing)
            print(f"Captured {count}/{len(pending)}: {record['teacher_status']}", flush=True)

    processed = [existing[int(row["dataset_index"])] for row in selected if int(row["dataset_index"]) in existing]
    report = {
        "source": str(args.input.resolve()), "selected": len(selected), "new_teacher_calls": len(pending),
        "resumed": len(selected) - len(pending), "structured_mode": args.structured_mode,
        "retry_failed": args.retry_failed, "teacher_status": dict(Counter(row["teacher_status"] for row in processed)),
        "teacher_is_ground_truth": False, "dpo_pairs_created": False,
        "next_step": "Analyze raw messages and reasoning before creating any DPO preference pairs.",
    }
    (output_dir / REPORT_NAME).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
