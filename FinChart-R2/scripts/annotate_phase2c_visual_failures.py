"""Annotate Phase 2C SFT visual/counting failures with a vision teacher.

Only the selected review queue is sent to the provider. The script does not
read, print, or persist API credentials. It writes resumable JSONL/JSON outputs
under results/phase2c_visual_diagnosis, which are ignored by Git.

Supported providers:
  - gemini: native image + JSON-schema response + normalized 0–1000 bbox.
  - openai_compatible: Chat Completions with JSON-object mode.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "phase2c_visual_diagnosis"
DEFAULT_QUEUE = RESULTS / "phase2c_visual_candidates.jsonl"
ANNOTATIONS = RESULTS / "phase2c_teacher_annotations.jsonl"
AUDIT_QUEUE = RESULTS / "phase2c_teacher_audit.jsonl"
REPORT = RESULTS / "phase2c_teacher_report.json"

ERROR_TYPES = {"VISUAL_GROUNDING", "COUNTING", "NON_VISUAL_OR_UNCLEAR"}
SUBTYPES = {
    "WRONG_SERIES",
    "WRONG_COLOR",
    "WRONG_CATEGORY",
    "WRONG_VALUE",
    "WRONG_POINT",
    "LEGEND_ASSOCIATION",
    "AXIS_ALIGNMENT",
    "COUNTING_ERROR",
    "EXTREMA_ERROR",
    "CROP_SMALL_TEXT",
    "OTHER_VISUAL",
}

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "error_type": {"type": "string", "enum": sorted(ERROR_TYPES)},
        "subtype": {"type": "string", "enum": sorted(SUBTYPES)},
        "target_series": {"type": ["string", "null"]},
        "target_category": {"type": ["string", "null"]},
        "target_color": {"type": ["string", "null"]},
        "relevant_value": {"type": ["string", "number", "null"]},
        "bbox": {
            "type": ["array", "null"],
            "items": {"type": "integer", "minimum": 0, "maximum": 1000},
            "minItems": 4,
            "maxItems": 4,
        },
        "evidence": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "needs_manual_review": {"type": "boolean"},
    },
    "required": [
        "error_type",
        "subtype",
        "target_series",
        "target_category",
        "target_color",
        "relevant_value",
        "bbox",
        "evidence",
        "confidence",
        "needs_manual_review",
    ],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write a separate annotation run without overwriting the default Phase 2C results.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent teacher requests; use a small value to respect provider limits.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--revalidate-existing",
        action="store_true",
        help="Re-run local schema/normalization validation on saved labels without API calls.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_annotation_checkpoint(rows: dict[int, dict[str, Any]]) -> None:
    """Atomically replace the resumable annotation index after each request."""
    ANNOTATIONS.parent.mkdir(parents=True, exist_ok=True)
    temporary = ANNOTATIONS.with_suffix(".tmp")
    temporary.write_text(
        "".join(
            json.dumps(rows[index], ensure_ascii=False, default=str) + "\n"
            for index in sorted(rows)
        ),
        encoding="utf-8",
    )
    temporary.replace(ANNOTATIONS)


def image_as_base64(image: Any) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def image_sha256(image: Any) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def env_value(primary: str, fallback: str = "") -> str:
    return os.getenv(primary, "").strip() or os.getenv(fallback, "").strip()


def teacher_config() -> dict[str, Any]:
    # Intentionally load only .env, never a secret-bearing .env.example fallback.
    load_dotenv(ROOT / ".env")
    provider = env_value("VISUAL_TEACHER_PROVIDER").lower() or "openai_compatible"
    key = env_value("VISUAL_TEACHER_API_KEY", "TEACHER_API_KEY")
    model = env_value("VISUAL_TEACHER_MODEL", "TEACHER_MODEL")
    base_url = env_value("VISUAL_TEACHER_BASE_URL", "TEACHER_BASE_URL")
    limit_text = env_value("VISUAL_TEACHER_LIMIT") or "100"
    confidence_text = env_value("VISUAL_TEACHER_MIN_CONFIDENCE") or "0.80"
    attempts_text = env_value("VISUAL_TEACHER_MAX_ATTEMPTS") or "2"
    if provider not in {"gemini", "openai_compatible"}:
        raise ValueError("VISUAL_TEACHER_PROVIDER must be gemini or openai_compatible")
    if not key or not model:
        raise RuntimeError("Visual teacher key/model missing. Set VISUAL_TEACHER_* in .env.")
    if provider == "openai_compatible" and "api.deepseek.com" in base_url.lower():
        raise RuntimeError(
            "Direct DeepSeek API is text-only for this workflow. Phase 2C sends chart images; "
            "configure a vision-capable OpenAI-compatible endpoint or Gemini instead."
        )
    try:
        limit, min_confidence, max_attempts = int(limit_text), float(confidence_text), int(attempts_text)
    except ValueError as exc:
        raise ValueError("VISUAL_TEACHER_LIMIT or VISUAL_TEACHER_MIN_CONFIDENCE is invalid") from exc
    if limit < 1 or max_attempts < 1 or not 0 <= min_confidence <= 1:
        raise ValueError("Visual teacher limit/confidence is outside its valid range")
    return {
        "provider": provider,
        "key": key,
        "model": model,
        "base_url": base_url,
        "limit": limit,
        "min_confidence": min_confidence,
        "max_attempts": max_attempts,
    }


def annotation_prompt(row: dict[str, Any]) -> str:
    return f"""You are an offline chart-grounding diagnosis teacher. Diagnose why the SFT prediction is wrong.
Return one JSON object that conforms exactly to the supplied schema; do not add markdown.

Every field is mandatory, including fields whose value is null. Return exactly these keys:
error_type, subtype, target_series, target_category, target_color, relevant_value,
bbox, evidence, confidence, needs_manual_review.
Choose error_type from: VISUAL_GROUNDING, COUNTING, NON_VISUAL_OR_UNCLEAR.
Choose subtype from: WRONG_SERIES, WRONG_COLOR, WRONG_CATEGORY, WRONG_VALUE,
WRONG_POINT, LEGEND_ASSOCIATION, AXIS_ALIGNMENT, COUNTING_ERROR, EXTREMA_ERROR,
CROP_SMALL_TEXT, OTHER_VISUAL. If error_type is COUNTING, subtype must be
COUNTING_ERROR. For unknown target values use null, never omit a key.

Use error_type VISUAL_GROUNDING when the failure is series/color/category/value/legend/axis/point/extrema/text localization.
Use COUNTING only when the failure is counting chart elements or intersections.
Use NON_VISUAL_OR_UNCLEAR if the failure is arithmetic, label ambiguity, or cannot be localized visually.
The bbox refers to the single most relevant chart evidence and must be [ymin, xmin, ymax, xmax] normalized 0–1000. Use null only if no defensible local region exists. Do not invent a bounding box.

Question: {row['question']}
ChartQA ground-truth answer: {row['ground_truth']}
Base prediction: {row['base_prediction']}
SFT prediction: {row['sft_prediction']}
Deterministic transition: {row['transition_tag']}
Phase 1 final verdict: {row.get('final_verdict')}
Phase 1 error type (if audited): {row.get('judge_error_type')}
Phase 1 judge evidence (if available): {row.get('judge_reason')}
Heuristic proposed subtype: {row.get('proposed_subtype')}
Heuristic location hint: {row.get('proposed_where')}
Manual audit feedback (re-evaluate independently; do not blindly follow it): {row.get('manual_audit_note', 'None')}

Before answering, check that your subtype agrees with your evidence and that any stated count/calculation is numerically consistent. If the ChartQA ground truth conflicts with what is visibly shown, use NON_VISUAL_OR_UNCLEAR and set needs_manual_review=true rather than forcing a visual label."""


def response_text_from_gemini(data: dict[str, Any]) -> str:
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("Gemini response has no candidates")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(str(part.get("text", "")) for part in parts)
    if not text:
        raise ValueError("Gemini response has no text payload")
    return text


def call_gemini(row: dict[str, Any], image: Any, config: dict[str, Any]) -> dict[str, Any]:
    base_url = config["base_url"] or "https://generativelanguage.googleapis.com/v1beta"
    url = f"{base_url.rstrip('/')}/models/{config['model']}:generateContent"
    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": annotation_prompt(row)},
                {"inlineData": {"mimeType": "image/png", "data": image_as_base64(image)}},
            ],
        }],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseJsonSchema": SCHEMA,
        },
    }
    response = requests.post(
        url,
        headers={"x-goog-api-key": config["key"], "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return json.loads(response_text_from_gemini(response.json()))


def call_openai_compatible(row: dict[str, Any], image: Any, config: dict[str, Any]) -> dict[str, Any]:
    if not config["base_url"]:
        raise RuntimeError("VISUAL_TEACHER_BASE_URL is required for openai_compatible")
    url = config["base_url"].rstrip("/")
    if not url.endswith("chat/completions"):
        url += "/chat/completions"
    payload = {
        "model": config["model"],
        "temperature": 0,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": annotation_prompt(row)},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + image_as_base64(image)},
                },
            ],
        }],
    }
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {config['key']}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        preview = str(content).replace("\n", " ").strip()[:400]
        raise ValueError(f"OpenAI-compatible response is not JSON: {preview!r}") from exc


def call_teacher(row: dict[str, Any], image: Any, config: dict[str, Any]) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(config["max_attempts"]):
        try:
            if config["provider"] == "gemini":
                return call_gemini(row, image, config)
            return call_openai_compatible(row, image, config)
        except Exception as exc:
            last_error = exc
            if attempt < config["max_attempts"] - 1:
                time.sleep(2**attempt)
    detail = str(last_error).replace("\n", " ")[:400] if last_error else "unknown error"
    raise RuntimeError(f"Teacher request failed after retries ({type(last_error).__name__}): {detail}")


def validate_annotation(payload: Any, min_confidence: float) -> tuple[dict[str, Any], list[str], str]:
    if not isinstance(payload, dict):
        return {}, ["NOT_OBJECT"], "INVALID_SCHEMA"
    payload = dict(payload)
    missing = set(SCHEMA["required"]) - set(payload)
    errors = ["MISSING:" + ",".join(sorted(missing))] if missing else []
    normalization_flags: list[str] = []
    if payload.get("error_type") not in ERROR_TYPES:
        errors.append("INVALID_ERROR_TYPE")
    if payload.get("subtype") not in SUBTYPES:
        errors.append("INVALID_SUBTYPE")
    elif payload.get("error_type") == "COUNTING" and payload.get("subtype") != "COUNTING_ERROR":
        errors.append("COUNTING_REQUIRES_COUNTING_ERROR_SUBTYPE")
    elif payload.get("error_type") == "VISUAL_GROUNDING" and payload.get("subtype") == "COUNTING_ERROR":
        errors.append("VISUAL_GROUNDING_CANNOT_USE_COUNTING_ERROR_SUBTYPE")
    bbox = payload.get("bbox")
    if bbox is not None:
        # Some OpenAI-compatible vision models return normalized coordinates in
        # [0, 1] despite the prompt requesting Gemini-style [0, 1000]. This
        # conversion is deterministic and preserved as an audit flag.
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in bbox)
            and all(0 <= value <= 1 for value in bbox)
            and any(not isinstance(value, int) for value in bbox)
        ):
            bbox = [round(value * 1000) for value in bbox]
            payload["bbox"] = bbox
            normalization_flags.append("BBOX_RESCALED_FROM_UNIT_INTERVAL")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(x, int) for x in bbox):
            errors.append("INVALID_BBOX_SHAPE")
        elif not all(0 <= x <= 1000 for x in bbox) or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            errors.append("INVALID_BBOX_RANGE")
    try:
        confidence = float(payload.get("confidence"))
        if not 0 <= confidence <= 1:
            errors.append("INVALID_CONFIDENCE_RANGE")
    except (TypeError, ValueError):
        confidence = 0.0
        errors.append("INVALID_CONFIDENCE")
    if not isinstance(payload.get("evidence"), str) or not payload.get("evidence", "").strip():
        errors.append("EMPTY_EVIDENCE")
    if not isinstance(payload.get("needs_manual_review"), bool):
        errors.append("INVALID_REVIEW_FLAG")
    if errors:
        return payload, normalization_flags + errors, "INVALID_SCHEMA"
    if confidence < min_confidence:
        return payload, normalization_flags + ["LOW_CONFIDENCE"], "LOW_CONFIDENCE"
    if payload["error_type"] == "NON_VISUAL_OR_UNCLEAR" or payload["needs_manual_review"]:
        return payload, normalization_flags, "REVIEW_REQUIRED"
    return payload, normalization_flags, "VALIDATED_CANDIDATE"


def stratified_audit(rows: list[dict[str, Any]], fraction: float = 0.20) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("annotation_status") == "VALIDATED_CANDIDATE":
            groups[str(row.get("teacher_label", {}).get("subtype", "OTHER_VISUAL"))].append(row)
    selected: list[dict[str, Any]] = []
    for subtype, group in sorted(groups.items()):
        group.sort(key=lambda row: int(row["dataset_index"]))
        selected.extend(group[: max(1, math.ceil(len(group) * fraction))])
    return selected


def annotate_one(row: dict[str, Any], image: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Call the teacher once and return a fully validated, resumable record."""
    try:
        payload = call_teacher(row, image, config)
        label, flags, status = validate_annotation(payload, config["min_confidence"])
        teacher_error = None
    except Exception as exc:
        label, flags, status = {}, [f"TEACHER_ERROR:{type(exc).__name__}"], "TEACHER_ERROR"
        teacher_error = str(exc).replace("\n", " ")[:500]
    return {
        **row,
        "image_sha256": image_sha256(image),
        "teacher_provider": config["provider"],
        "teacher_model": config["model"],
        "teacher_label": label,
        "annotation_status": status,
        "validation_flags": flags,
        "teacher_error": teacher_error,
        "manual_audit_status": "PENDING" if status == "VALIDATED_CANDIDATE" else "NOT_ELIGIBLE",
    }


def main() -> None:
    global RESULTS, DEFAULT_QUEUE, ANNOTATIONS, AUDIT_QUEUE, REPORT
    args = parse_args()
    if args.output_dir is not None:
        RESULTS = args.output_dir.resolve()
        DEFAULT_QUEUE = RESULTS / "phase2c_visual_candidates.jsonl"
        ANNOTATIONS = RESULTS / "phase2c_teacher_annotations.jsonl"
        AUDIT_QUEUE = RESULTS / "phase2c_teacher_audit.jsonl"
        REPORT = RESULTS / "phase2c_teacher_report.json"
    queue = load_jsonl(args.queue)
    if not queue:
        raise RuntimeError("Phase 2C queue is empty")

    if args.dry_run:
        selected = queue[: args.limit or 100]
        print(json.dumps({"dry_run": True, "queue_size": len(queue), "selected": len(selected)}, indent=2))
        return

    config = teacher_config()
    if args.workers < 1 or args.workers > 4:
        raise ValueError("--workers must be between 1 and 4")
    if args.revalidate_existing:
        if not ANNOTATIONS.exists():
            raise FileNotFoundError("No existing Phase 2C annotations to revalidate")
        output_rows = {int(row["dataset_index"]): row for row in load_jsonl(ANNOTATIONS)}
        for index, record in output_rows.items():
            label, flags, status = validate_annotation(record.get("teacher_label", {}), config["min_confidence"])
            record["teacher_label"] = label
            record["annotation_status"] = status
            record["validation_flags"] = flags
            if status != "TEACHER_ERROR":
                record["teacher_error"] = None
            output_rows[index] = record
        write_annotation_checkpoint(output_rows)
        processed = [output_rows[index] for index in sorted(output_rows)]
        audit = stratified_audit(processed)
        AUDIT_QUEUE.write_text(
            "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in audit),
            encoding="utf-8",
        )
        report = {
            "selected_candidates": len(processed),
            "new_teacher_calls": 0,
            "resumed_annotations": len(processed),
            "revalidated_existing": True,
            "annotation_status": dict(Counter(row["annotation_status"] for row in processed)),
            "teacher_subtypes": dict(Counter(row.get("teacher_label", {}).get("subtype", "UNLABELED") for row in processed)),
            "manual_audit_queue": len(audit),
            "manual_audit_fraction_target": 0.20,
            "teacher_is_ground_truth": False,
            "training_export_created": False,
        }
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    limit = args.limit or config["limit"]
    existing = {} if args.no_resume or not ANNOTATIONS.exists() else {
        int(row["dataset_index"]): row for row in load_jsonl(ANNOTATIONS)
    }
    priority = sorted(
        queue,
        key=lambda row: (
            row.get("candidate_source") != "PHASE1_CONFIRMED_VISUAL_OR_COUNTING",
            row.get("transition_tag") != "BOTH_WRONG",
            int(row["dataset_index"]),
        ),
    )
    selected = priority[:limit]
    pending = [row for row in selected if int(row["dataset_index"]) not in existing]
    dataset = load_dataset("HuggingFaceM4/ChartQA", split="val")
    output_rows = dict(existing)

    work_items: list[tuple[dict[str, Any], Any]] = []
    for row in pending:
        index = int(row["dataset_index"])
        example = dataset[index]
        source_question = str(example.get("query", example.get("question", ""))).strip()
        if source_question != str(row["question"]).strip():
            raise ValueError(f"Question mismatch for dataset_index={index}")
        work_items.append((row, example["image"]))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(annotate_one, row, image, config) for row, image in work_items]
        for position, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            output_rows[int(record["dataset_index"])] = record
            write_annotation_checkpoint(output_rows)
            if position % 10 == 0 or position == len(pending):
                print(f"Annotated {position}/{len(pending)}", flush=True)

    # A final write also handles a resumed run with no pending records.
    write_annotation_checkpoint(output_rows)

    processed = [output_rows[int(row["dataset_index"])] for row in selected if int(row["dataset_index"]) in output_rows]
    audit = stratified_audit(processed)
    AUDIT_QUEUE.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in audit),
        encoding="utf-8",
    )
    report = {
        "selected_candidates": len(selected),
        "new_teacher_calls": len(pending),
        "resumed_annotations": len(selected) - len(pending),
        "annotation_status": dict(Counter(row["annotation_status"] for row in processed)),
        "teacher_subtypes": dict(
            Counter(
                row.get("teacher_label", {}).get("subtype", "UNLABELED") for row in processed
            )
        ),
        "manual_audit_queue": len(audit),
        "manual_audit_fraction_target": 0.20,
        "teacher_is_ground_truth": False,
        "training_export_created": False,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
