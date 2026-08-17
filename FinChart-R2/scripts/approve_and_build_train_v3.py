"""Create the manually approved, strict-clean Phase 2A pilot SFT dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
V2 = RESULTS / "checkpoints" / "phase2a_teacher_annotations_500_v2.jsonl"
OUT_JSONL = RESULTS / "phase2a_pilot_500_v3_train_clean.jsonl"
OUT_CSV = RESULTS / "phase2a_pilot_500_v3_train_clean_metadata.csv"
OUT_REPORT = RESULTS / "phase2a_pilot_500_v3_train_clean_report.json"
OUT_APPROVAL = RESULTS / "phase2a_pilot_500_v3_manual_audit_approval.json"

TARGET_SHARE = {
    "numerical_reasoning": 0.45,
    "visual_grounding": 0.30,
    "counting": 0.15,
    "logical_reasoning": 0.10,
}
ALLOWED_RELATIONS = {"EXACT_MATCH", "NUMERIC_MATCH", "NUMERIC_SEQUENCE_MATCH"}


def load_rows() -> list[dict]:
    return [json.loads(line) for line in V2.read_text(encoding="utf-8").splitlines() if line.strip()]


def incompatible_task_operation(row: pd.Series) -> bool:
    task, operation = row["teacher_task_type"], row["operation"]
    # These are unambiguous mismatches; other combinations remain eligible so
    # direct lookup and multi-step chart questions are not discarded blindly.
    if task == "counting":
        return operation != "count"
    if task == "visual_grounding":
        return operation in {"sum", "average", "median", "ratio", "percentage", "percentage_change", "count"}
    if task == "logical_reasoning":
        return operation in {"sum", "average", "median", "ratio", "percentage", "percentage_change", "count"}
    if task == "numerical_reasoning":
        return operation == "count"
    return True


def make_target(row: pd.Series) -> str:
    lines: list[str] = []
    if row.get("target_series"):
        lines.append(f"Target series: {row['target_series']}")
    if row.get("target_category"):
        lines.append(f"Target category: {row['target_category']}")
    values = row.get("relevant_values")
    if isinstance(values, list) and values:
        lines.append("Relevant values: " + ", ".join(map(str, values)))
    if row.get("operation") and row["operation"] != "none":
        lines.append(f"Operation: {row['operation']}")
    if row.get("calculation"):
        lines.append(f"Calculation: {row['calculation']}")
    lines.append(f"Answer: {row['dataset_answer']}")
    return "\n".join(lines)


def main() -> None:
    rows = load_rows()
    if len(rows) != 500:
        raise RuntimeError("Expected 500 v2 records.")
    frame = pd.DataFrame(rows)
    reasons = pd.Series("KEEP", index=frame.index, dtype="object")
    reasons.loc[frame.annotation_status != "VALIDATED"] = "NOT_VALIDATED"
    reasons.loc[(reasons == "KEEP") & (frame.teacher_confidence < 0.90)] = "CONFIDENCE_LT_0_90"
    reasons.loc[(reasons == "KEEP") & ~frame.answer_relation.isin(ALLOWED_RELATIONS)] = "ANSWER_RELATION_NOT_CLEAN"
    reasons.loc[(reasons == "KEEP") & frame.apply(incompatible_task_operation, axis=1)] = "TASK_OPERATION_INCOMPATIBLE"
    clean = frame[reasons == "KEEP"].copy()
    rejected = frame.assign(cleaning_decision=reasons)[reasons != "KEEP"].copy()
    if len(clean) < 350:
        raise RuntimeError(f"Strict-clean pool unexpectedly small: {len(clean)}")

    shares = clean.teacher_task_type.value_counts(normalize=True).to_dict()
    clean["curriculum_weight"] = clean.teacher_task_type.map(lambda task: TARGET_SHARE[task] / shares[task])
    clean["sft_target"] = clean.apply(make_target, axis=1)
    clean["image_locator"] = clean.dataset_index.map(lambda index: {"dataset": "HuggingFaceM4/ChartQA", "split": "train", "dataset_index": int(index)})
    clean["user_prompt"] = clean.question.map(lambda question: f"Look carefully at the chart and answer the question.\n\nQuestion: {question}")

    export_columns = [
        "sample_id", "dataset_index", "image_locator", "user_prompt", "question", "dataset_answer",
        "teacher_task_type", "teacher_subtype", "target_series", "target_category", "relevant_values",
        "operation", "calculation", "teacher_confidence", "answer_relation", "arithmetic_valid",
        "validation_flags", "curriculum_weight", "sft_target",
    ]
    records = clean[export_columns].to_dict("records")
    with OUT_JSONL.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    clean[export_columns].to_csv(OUT_CSV, index=False)

    approval = {
        "approval_scope": "Phase 2A 500-sample pilot only",
        "approval_source": "user-authorized manual audit decision",
        "manual_audit_approved": True,
        "source_checkpoint": str(V2),
        "approved_for": "pilot QLoRA SFT experiment",
        "not_approved_for": "full production SFT dataset or Phase 2A final completion",
        "strict_cleaning_rules": [
            "annotation_status == VALIDATED",
            "teacher_confidence >= 0.90",
            "answer_relation in EXACT_MATCH, NUMERIC_MATCH, NUMERIC_SEQUENCE_MATCH",
            "no unambiguous task-operation incompatibility",
        ],
    }
    OUT_APPROVAL.write_text(json.dumps(approval, indent=2), encoding="utf-8")
    report = {
        "source_records": len(frame),
        "strict_clean_records": len(clean),
        "strict_clean_rate": len(clean) / len(frame),
        "rejected_by_reason": Counter(reasons[reasons != "KEEP"]),
        "task_distribution": Counter(clean.teacher_task_type),
        "sampling_weights": {task: TARGET_SHARE[task] / shares[task] for task in TARGET_SHARE},
        "manual_audit_approved": True,
        "phase1_validation_used_for_training": False,
        "status": "APPROVED_FOR_PILOT_SFT",
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(report, indent=2, default=dict))


if __name__ == "__main__":
    main()
