"""Profile provisional Phase 2C DPO pairs with deterministic heuristic labels.

The exported teacher pairs do not contain an audited task_type. This script adds no
labels to the training data; it writes an aggregate diagnostic report for dataset
balancing and README reporting. The taxonomy is intentionally transparent:

1. yes/no ground truth -> logical_reasoning
2. explicit count intent -> counting
3. arithmetic intent -> numerical_reasoning
4. remaining lookup/color/category/value intent -> visual_grounding

These categories are heuristics and must not be presented as ground truth.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results"
    / "finchart_r2_phase2c_train_mining"
    / "phase2c_teacher_v1_dpo_candidates_provisional.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.with_name(
    "phase2c_teacher_v1_dpo_candidates_diversity_report.json"
)

OPERATION_LINE = re.compile(r"^\s*Operation\s*:\s*(.+?)\s*$", re.I | re.M)
NUMERICAL_INTENT = re.compile(
    r"\b(sum|add(?:ition)?|plus|subtract(?:ion)?|difference|average|mean|"
    r"median|ratio|divide|division|divided|multiply|multiplication|product|"
    r"half|twice|percentage change|percent change|increase|decrease)\b",
    re.I,
)
COUNT_INTENT = re.compile(
    r"\b(how many|number of|for how many|in how many|count)\b", re.I
)
RATIO_COUNT_PHRASE = re.compile(r"\bhow many times\b", re.I)
SAMPLE_PROJECTION = re.compile(
    r"\b(?:ask|survey|sample)\b.*\bhow many\b|"
    r"\bhow many\b.*\b(?:people|respondents)\b.*\b(?:will|would)\b",
    re.I,
)

OPERATION_PATTERNS = (
    ("count", re.compile(r"\bcount\b", re.I)),
    ("average", re.compile(r"\b(?:average|mean)\b", re.I)),
    ("median", re.compile(r"\bmedian\b", re.I)),
    ("sum", re.compile(r"\b(?:sum|add|addition|plus|total)\b", re.I)),
    (
        "difference",
        re.compile(r"\b(?:subtract|subtraction|difference|deduct)\b", re.I),
    ),
    ("ratio", re.compile(r"\b(?:ratio|divide|division|divided)\b", re.I)),
    (
        "product",
        re.compile(r"\b(?:multiply|multiplication|product|times)\b", re.I),
    ),
    (
        "comparison",
        re.compile(
            r"\b(?:compare|comparison|greater|less|equal|check|vs\.?|versus)\b",
            re.I,
        ),
    ),
    (
        "extrema",
        re.compile(
            r"\b(?:minimum|maximum|min|max|smallest|largest|highest|lowest|rank)\b",
            re.I,
        ),
    ),
    ("lookup", re.compile(r"\b(?:identify|locate|match|find|read)\b", re.I)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def answer_type(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"yes", "no"}:
        return "yes_no"
    try:
        float(text.replace(",", "").removesuffix("%"))
    except ValueError:
        return "text"
    return "numeric"


def raw_operation(row: dict[str, Any]) -> str:
    match = OPERATION_LINE.search(str(row.get("chosen", "")))
    return match.group(1).strip() if match else ""


def task_category(row: dict[str, Any]) -> str:
    question = str(row.get("prompt", ""))
    operation = raw_operation(row)

    if answer_type(row.get("ground_truth")) == "yes_no":
        return "logical_reasoning"
    if re.search(r"\bcount\b", operation, re.I):
        return "counting"
    if (
        COUNT_INTENT.search(question)
        and not RATIO_COUNT_PHRASE.search(question)
        and not SAMPLE_PROJECTION.search(question)
    ):
        return "counting"
    if NUMERICAL_INTENT.search(operation) or NUMERICAL_INTENT.search(question):
        return "numerical_reasoning"
    return "visual_grounding"


def operation_category(row: dict[str, Any]) -> str:
    operation = raw_operation(row)
    flags = [name for name, pattern in OPERATION_PATTERNS if pattern.search(operation)]
    arithmetic = {
        name
        for name in flags
        if name in {"average", "median", "sum", "difference", "ratio", "product"}
    }
    if len(arithmetic) >= 2 or ("comparison" in flags and arithmetic):
        return "multi_step"
    for name in (
        "count",
        "average",
        "median",
        "sum",
        "difference",
        "ratio",
        "product",
        "comparison",
        "extrema",
        "lookup",
    ):
        if name in flags:
            return name
    return "other"


def distribution(counter: Counter[str], total: int) -> dict[str, dict[str, float | int]]:
    return {
        key: {"count": count, "percentage": round(100.0 * count / total, 1)}
        for key, count in counter.most_common()
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input.resolve())
    if not rows:
        raise ValueError("Candidate input is empty")

    required = {
        "dataset_index",
        "image_split",
        "image_index",
        "prompt",
        "chosen",
        "rejected",
        "ground_truth",
    }
    for line_number, row in enumerate(rows, 1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Input line {line_number} missing {sorted(missing)}")

    total = len(rows)
    task_counts = Counter(task_category(row) for row in rows)
    operation_counts = Counter(operation_category(row) for row in rows)
    answer_counts = Counter(answer_type(row["ground_truth"]) for row in rows)
    dataset_indices = [int(row["dataset_index"]) for row in rows]
    image_refs = {(row["image_split"], int(row["image_index"])) for row in rows}

    report = {
        "source": str(args.input.resolve()),
        "records": total,
        "unique_dataset_indices": len(set(dataset_indices)),
        "unique_image_references": len(image_refs),
        "dataset_index_min": min(dataset_indices),
        "dataset_index_max": max(dataset_indices),
        "image_splits": dict(Counter(str(row["image_split"]) for row in rows)),
        "task_category_heuristic": distribution(task_counts, total),
        "operation_category_heuristic": distribution(operation_counts, total),
        "answer_type": distribution(answer_counts, total),
        "taxonomy_precedence": [
            "yes/no answer -> logical_reasoning",
            "explicit counting intent -> counting",
            "arithmetic intent -> numerical_reasoning",
            "remaining lookup/color/category/value intent -> visual_grounding",
        ],
        "labels_are_ground_truth": False,
        "manual_audit_required_before_dpo": True,
        "dpo_pilot_readiness": (
            "Enough unique train-only pairs for a small pilot after manual audit; "
            "not balanced or large enough for a final general-purpose DPO claim."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
