"""Repair and re-annotate only non-validated records from the Phase 2A pilot.

The v1 checkpoint is immutable. This command produces a v2 checkpoint and
exports only v2 VALIDATED records for the next audit/training decision.
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from datasets import load_dataset
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHECKPOINTS = RESULTS / "checkpoints"
AUDIT_DIR = RESULTS / "audit"
V1 = CHECKPOINTS / "phase2a_teacher_annotations_500.jsonl"
V2 = CHECKPOINTS / "phase2a_teacher_annotations_500_v2.jsonl"
SEED = 42

TASK_TYPES = {"visual_grounding", "numerical_reasoning", "counting", "logical_reasoning"}
SUBTYPES = {
    "direct_value", "temporal_lookup", "position", "series_or_legend", "color", "extrema", "intersection", "category_lookup",
    "sum", "difference", "average", "median", "ratio", "percentage", "percentage_change", "max_difference", "min_max_arithmetic", "multi_step",
    "count_elements", "count_threshold", "count_occurrences", "count_intersections",
    "boolean_comparison", "ranking", "trend", "conditional", "comparison",
}
OPERATIONS = {"none", "lookup", "sum", "difference", "average", "median", "ratio", "percentage", "percentage_change", "count", "comparison", "max_difference", "min_max", "multi_step"}

TASK_FIXES = {}
SUBTYPE_FIXES = {
    "bolean_comparison": "boolean_comparison",
    "count_occurences": "count_occurrences",
}
OPERATION_FIXES = {
    "diference": "difference",
    "max_diference": "max_difference",
    "max_min": "min_max",
}
NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    text = re.sub(r"^\s*(?:answer|final answer|result)\s*:\s*", "", str(value), flags=re.I)
    return re.sub(r"\s+", " ", text).strip(" \t\n\r.,;:")


def numeric_values(value: Any) -> list[float]:
    text = normalize_text(value).replace("%", "")
    return [float(x.replace(",", "")) for x in NUMBER_RE.findall(text)]


def answer_relation(dataset_answer: Any, teacher_answer: Any, tol: float = 1e-5) -> str:
    left, right = normalize_text(dataset_answer).lower(), normalize_text(teacher_answer).lower()
    if left == right:
        return "EXACT_MATCH"
    a, b = numeric_values(left), numeric_values(right)
    if len(a) == len(b) and a and all(abs(x - y) <= tol for x, y in zip(a, b)):
        return "NUMERIC_SEQUENCE_MATCH" if len(a) > 1 else "NUMERIC_MATCH"
    if len(a) == len(b) == 1 and (abs(a[0] * 100 - b[0]) <= tol or abs(b[0] * 100 - a[0]) <= tol):
        return "PERCENT_PROPORTION_EQUIVALENT"
    return "CONFLICT"


def repair_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    fixed = dict(payload)
    actions: list[str] = []
    for key, mapping in (("task_type", TASK_FIXES), ("subtype", SUBTYPE_FIXES), ("operation", OPERATION_FIXES)):
        original = str(fixed.get(key, "")).strip().lower()
        replacement = mapping.get(original, original)
        if replacement != original:
            actions.append(f"CANONICAL_{key.upper()}:{original}->{replacement}")
        fixed[key] = replacement
    return fixed, actions


def validate_schema(payload: dict[str, Any]) -> list[str]:
    required = {"task_type", "subtype", "target_series", "target_category", "relevant_values", "operation", "calculation", "final_answer", "confidence"}
    errors: list[str] = []
    missing = required - set(payload)
    if missing:
        errors.append("MISSING_KEYS:" + ",".join(sorted(missing)))
    if payload.get("task_type") not in TASK_TYPES:
        errors.append("INVALID_TASK_TYPE")
    if payload.get("subtype") not in SUBTYPES:
        errors.append("INVALID_SUBTYPE")
    if payload.get("operation") not in OPERATIONS:
        errors.append("INVALID_OPERATION")
    if not isinstance(payload.get("relevant_values"), list):
        errors.append("RELEVANT_VALUES_NOT_LIST")
    try:
        if not 0 <= float(payload.get("confidence")) <= 1:
            errors.append("INVALID_CONFIDENCE_RANGE")
    except (TypeError, ValueError):
        errors.append("INVALID_CONFIDENCE")
    if not normalize_text(payload.get("final_answer")):
        errors.append("EMPTY_FINAL_ANSWER")
    return errors


def scalar_values(values: Any) -> list[float] | None:
    if not isinstance(values, list) or not values:
        return None
    out: list[float] = []
    for item in values:
        if isinstance(item, (dict, list)):
            return None
        parsed = numeric_values(item)
        if len(parsed) != 1:
            return None
        out.append(parsed[0])
    return out


def recompute_if_unambiguous(payload: dict[str, Any]) -> bool | None:
    """Return None when the schema lacks sufficient structure for safe math checks."""
    values = scalar_values(payload.get("relevant_values"))
    answer = numeric_values(payload.get("final_answer"))
    op, subtype = payload.get("operation"), payload.get("subtype")
    if values is None or len(answer) != 1:
        return None
    expected: float | None = None
    if op == "sum" and subtype == "sum": expected = sum(values)
    elif op == "average" and subtype == "average": expected = sum(values) / len(values)
    elif op == "median" and subtype == "median": expected = statistics.median(values)
    elif op == "difference" and subtype == "difference" and len(values) == 2: expected = values[0] - values[1]
    elif op == "ratio" and subtype == "ratio" and len(values) == 2 and values[1] != 0: expected = values[0] / values[1]
    elif op == "percentage" and subtype == "percentage" and len(values) == 2 and values[1] != 0: expected = values[0] / values[1] * 100
    if expected is None:
        return None
    return abs(expected - answer[0]) <= 1e-4


def assess(base: dict[str, Any], teacher: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    teacher, actions = repair_payload(teacher)
    errors = validate_schema(teacher)
    if errors:
        return teacher, {"annotation_status": "INVALID_SCHEMA", "answer_relation": None, "arithmetic_valid": None, "validation_flags": actions + errors}
    if float(teacher["confidence"]) < 0.80:
        return teacher, {"annotation_status": "LOW_CONFIDENCE", "answer_relation": None, "arithmetic_valid": None, "validation_flags": actions + ["LOW_TEACHER_CONFIDENCE"]}
    relation = answer_relation(base["dataset_answer"], teacher["final_answer"])
    arithmetic = recompute_if_unambiguous(teacher)
    if arithmetic is False:
        return teacher, {"annotation_status": "REVIEW_CONFLICT", "answer_relation": relation, "arithmetic_valid": False, "validation_flags": actions + ["ARITHMETIC_RECOMPUTE_FAILED"]}
    if relation in {"EXACT_MATCH", "NUMERIC_MATCH", "NUMERIC_SEQUENCE_MATCH"}:
        status = "VALIDATED"
    elif relation == "PERCENT_PROPORTION_EQUIVALENT":
        status = "REVIEW_REPRESENTATION"
    else:
        status = "REVIEW_CONFLICT"
    return teacher, {"annotation_status": status, "answer_relation": relation, "arithmetic_valid": arithmetic, "validation_flags": actions}


def image_url(image: Any) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def extract_json(text: str) -> dict[str, Any]:
    text = str(text).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object in teacher response")
    parsed, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(parsed, dict):
        raise ValueError("Teacher JSON is not an object")
    return parsed


def teacher_prompt(base: dict[str, Any]) -> str:
    return f'''Return only one valid JSON object. Use exact enum spellings.
task_type: {sorted(TASK_TYPES)}
subtype: {sorted(SUBTYPES)}
operation: {sorted(OPERATIONS)}
Schema: {{"task_type":"...","subtype":"...","target_series":null,"target_category":null,"relevant_values":[],"operation":"none","calculation":null,"final_answer":"...","confidence":0.0}}
Rules: use only values required for the answer; for count include only qualifying values in relevant_values; for max_difference include the winning pair and put the period/category in target_category; return a bare numeric answer when the question asks for a number.
Question: {base["question"]}
Dataset answer: {base["dataset_answer"]}'''


def call_teacher(base: dict[str, Any], image: Any, config: dict[str, str]) -> dict[str, Any]:
    payload = {"model": config["model"], "temperature": 0, "messages": [{"role": "system", "content": "You are a precise multimodal chart annotation teacher. Return JSON only."}, {"role": "user", "content": [{"type": "text", "text": teacher_prompt(base)}, {"type": "image_url", "image_url": {"url": image_url(image)}}]}]}
    url = config["base"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {config['key']}", "Content-Type": "application/json"}
    last: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            return extract_json(response.json()["choices"][0]["message"]["content"])
        except Exception as exc:
            last = exc
            print(f"Retry {attempt + 1}/3: {type(exc).__name__}", flush=True)
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(str(last))


def base_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in ("sample_id", "dataset_index", "question", "dataset_answer", "raw_answer", "image_present")}


def teacher_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {"task_type": row.get("teacher_task_type"), "subtype": row.get("teacher_subtype"), "target_series": row.get("target_series"), "target_category": row.get("target_category"), "relevant_values": row.get("relevant_values"), "operation": row.get("operation"), "calculation": row.get("calculation"), "final_answer": row.get("teacher_final_answer"), "confidence": row.get("teacher_confidence")}


def write_exports(rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    valid = frame[frame.annotation_status == "VALIDATED"].copy()
    weights = {"numerical_reasoning": .45, "visual_grounding": .30, "counting": .15, "logical_reasoning": .10}
    shares = valid.teacher_task_type.value_counts(normalize=True).to_dict()
    valid["curriculum_weight"] = valid.teacher_task_type.map(lambda task: weights.get(task, 0) / shares.get(task, 1))
    def target(row: pd.Series) -> str:
        lines = []
        for label, key in (("Target series", "target_series"), ("Target category", "target_category")):
            if row.get(key): lines.append(f"{label}: {row[key]}")
        if isinstance(row.relevant_values, list) and row.relevant_values: lines.append("Relevant values: " + ", ".join(map(str, row.relevant_values)))
        if row.operation and row.operation != "none": lines.append(f"Operation: {row.operation}")
        if row.calculation: lines.append(f"Calculation: {row.calculation}")
        lines.append(f"Answer: {row.dataset_answer}")
        return "\n".join(lines)
    valid["sft_target"] = valid.apply(target, axis=1)
    cols = ["sample_id", "dataset_index", "question", "dataset_answer", "teacher_task_type", "teacher_subtype", "target_series", "target_category", "relevant_values", "operation", "calculation", "teacher_confidence", "curriculum_weight", "sft_target"]
    records = valid[cols].to_dict("records")
    save_jsonl(records, RESULTS / "phase2a_pilot_500_v2_sft_candidate.jsonl")
    valid[cols].to_csv(RESULTS / "phase2a_pilot_500_v2_sft_candidate_metadata.csv", index=False)
    random.seed(SEED)
    audit = pd.concat([valid.groupby("teacher_task_type", group_keys=False).apply(lambda x: x.sample(min(15, len(x)), random_state=SEED)), frame[frame.annotation_status != "VALIDATED"].sample(min(40, len(frame[frame.annotation_status != "VALIDATED"])), random_state=SEED)], ignore_index=True)
    audit.to_csv(AUDIT_DIR / "phase2a_teacher_annotation_audit_v2.csv", index=False)
    report = {"processed_samples": len(frame), "validated_samples": len(valid), "validated_rate": len(valid) / len(frame), "annotation_status": Counter(frame.annotation_status), "task_distribution": Counter(valid.teacher_task_type), "manual_audit_approved": False, "phase1_validation_used_for_training": False, "status": "READY_FOR_MANUAL_AUDIT" if len(valid)/len(frame) >= .70 else "NEEDS_REVIEW"}
    (RESULTS / "phase2a_pilot_500_v2_report.json").write_text(json.dumps(report, indent=2, default=dict), encoding="utf-8")


def main() -> None:
    load_dotenv(ROOT / ".env")
    if not os.getenv("TEACHER_API_KEY"): load_dotenv(ROOT / ".env.example")
    config = {"key": os.getenv("TEACHER_API_KEY", ""), "base": os.getenv("TEACHER_BASE_URL", ""), "model": os.getenv("TEACHER_MODEL", "")}
    if not all(config.values()): raise RuntimeError("Teacher configuration missing")
    rows = load_jsonl(V1)
    if len(rows) != 500: raise RuntimeError("Expected the immutable 500-record v1 checkpoint")
    prior_v2 = {row["sample_id"]: row for row in load_jsonl(V2)}
    dataset = load_dataset("HuggingFaceM4/ChartQA", split="train")
    repaired: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for row in rows:
        base = base_from_row(row)
        teacher, validation = assess(base, teacher_from_row(row))
        row = {**row, "teacher_task_type": teacher.get("task_type"), "teacher_subtype": teacher.get("subtype"), "operation": teacher.get("operation"), "annotation_status": validation["annotation_status"], "answer_relation": validation["answer_relation"], "arithmetic_valid": validation["arithmetic_valid"], "validation_flags": validation["validation_flags"], "validation_version": "v2"}
        if row["annotation_status"] == "VALIDATED": repaired.append(row)
        else: pending.append(row)
    print(f"Recovered without teacher calls: {len(repaired)}; re-annotation queue: {len(pending)}", flush=True)
    all_rows = {row["sample_id"]: row for row in repaired}
    all_rows.update(prior_v2)
    for position, row in enumerate(pending, 1):
        if row["sample_id"] in prior_v2: continue
        base = base_from_row(row)
        try:
            teacher = call_teacher(base, dataset[int(base["dataset_index"])]["image"], config)
            teacher, validation = assess(base, teacher)
            row.update({"teacher_task_type": teacher.get("task_type"), "teacher_subtype": teacher.get("subtype"), "target_series": teacher.get("target_series"), "target_category": teacher.get("target_category"), "relevant_values": teacher.get("relevant_values"), "operation": teacher.get("operation"), "calculation": teacher.get("calculation"), "teacher_final_answer": teacher.get("final_answer"), "teacher_confidence": teacher.get("confidence"), **validation, "validation_version": "v2"})
        except Exception as exc:
            row.update({"annotation_status": "TEACHER_ERROR", "validation_flags": [f"TEACHER_ERROR:{type(exc).__name__}"], "validation_version": "v2"})
        all_rows[row["sample_id"]] = row
        if position % 10 == 0:
            save_jsonl([all_rows[k] for k in sorted(all_rows)], V2)
            print(f"Saved v2: {len(all_rows)}/500", flush=True)
    final_rows = [all_rows[row["sample_id"]] for row in rows]
    save_jsonl(final_rows, V2)
    write_exports(final_rows)
    print("Phase 2A v2 export complete.", flush=True)


if __name__ == "__main__":
    main()
