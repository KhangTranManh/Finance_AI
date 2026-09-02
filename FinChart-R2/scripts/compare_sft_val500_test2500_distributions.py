#!/usr/bin/env python3
"""Compare task, answer, and operation distributions on SFT val/test runs.

Labels are deterministic question-intent heuristics. They are suitable for
distribution diagnostics and teacher-queue routing, but are not ground-truth
error causes. ChartQA validation and test records remain evaluation-only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from analyze_sft_test2500_failures import (
    answer_type,
    operation_heuristic,
    task_heuristic,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_DIR
    / "results"
    / "vali"
    / "sft_vllm_chartqa_3000"
    / "sft_vllm_chartqa_val500_test2500_predictions.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "results" / "phase2d_sft_val500_test2500_distribution"
)
EXPECTED = {"val": 500, "test": 2500}
TASK_ORDER = (
    "numerical_reasoning",
    "counting",
    "visual_grounding",
    "logical_reasoning",
)
ANSWER_ORDER = ("numeric", "text", "yes_no")
OPERATION_ORDER = (
    "lookup",
    "count",
    "difference",
    "average",
    "sum",
    "extrema",
    "ratio",
    "percentage",
    "comparison",
    "color_lookup",
    "product",
    "median",
    "percentage_change",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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


def annotate(row: dict[str, Any]) -> dict[str, Any]:
    operation = operation_heuristic(str(row["question"]), row["ground_truth"])
    return {
        **row,
        "task_type_heuristic": task_heuristic(operation),
        "answer_type": answer_type(row["ground_truth"]),
        "operation_type_heuristic": operation,
        "labels_are_ground_truth": False,
        "evaluation_only": True,
        "allowed_for_training": False,
    }


def distribution(
    rows: list[dict[str, Any]],
    label: Callable[[dict[str, Any]], str],
    order: tuple[str, ...],
) -> dict[str, dict[str, float | int]]:
    total = len(rows)
    failures = [row for row in rows if not row["deterministic_correct"]]
    all_counts = Counter(label(row) for row in rows)
    failure_counts = Counter(label(row) for row in failures)
    names = list(order) + sorted((set(all_counts) | set(failure_counts)) - set(order))
    output: dict[str, dict[str, float | int]] = {}
    for name in names:
        count = all_counts[name]
        failed = failure_counts[name]
        output[name] = {
            "total": count,
            "total_share_pct": round(100 * count / total, 2) if total else 0.0,
            "failures": failed,
            "failure_share_pct": (
                round(100 * failed / len(failures), 2) if failures else 0.0
            ),
            "error_rate_pct": round(100 * failed / count, 2) if count else 0.0,
            "failure_overrepresentation_pp": round(
                (100 * failed / len(failures) if failures else 0.0)
                - (100 * count / total if total else 0.0),
                2,
            ),
        }
    return output


def split_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row for row in rows if not row["deterministic_correct"]]
    return {
        "total": len(rows),
        "correct": len(rows) - len(failures),
        "failures": len(failures),
        "accuracy_pct": round(100 * (len(rows) - len(failures)) / len(rows), 2),
        "task_type": distribution(
            rows, lambda row: row["task_type_heuristic"], TASK_ORDER
        ),
        "answer_type": distribution(rows, lambda row: row["answer_type"], ANSWER_ORDER),
        "operation_type": distribution(
            rows, lambda row: row["operation_type_heuristic"], OPERATION_ORDER
        ),
    }


def split_delta(
    val_report: dict[str, Any], test_report: dict[str, Any], section: str
) -> dict[str, dict[str, float]]:
    names = set(val_report[section]) | set(test_report[section])
    return {
        name: {
            "test_minus_val_total_share_pp": round(
                test_report[section].get(name, {}).get("total_share_pct", 0.0)
                - val_report[section].get(name, {}).get("total_share_pct", 0.0),
                2,
            ),
            "test_minus_val_failure_share_pp": round(
                test_report[section].get(name, {}).get("failure_share_pct", 0.0)
                - val_report[section].get(name, {}).get("failure_share_pct", 0.0),
                2,
            ),
            "test_minus_val_error_rate_pp": round(
                test_report[section].get(name, {}).get("error_rate_pct", 0.0)
                - val_report[section].get(name, {}).get("error_rate_pct", 0.0),
                2,
            ),
        }
        for name in sorted(names)
    }


def dominant(report: dict[str, Any], section: str, metric: str) -> str:
    return max(report[section], key=lambda name: report[section][name][metric])


def main() -> None:
    args = parse_args()
    source_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = read_jsonl(source_path)

    annotated: dict[str, list[dict[str, Any]]] = {}
    for split, expected in EXPECTED.items():
        split_rows = sorted(
            (annotate(row) for row in source_rows if row.get("split") == split),
            key=lambda row: int(row["dataset_index"]),
        )
        if len(split_rows) != expected:
            raise RuntimeError(f"Expected {expected} {split} rows, found {len(split_rows)}")
        if [int(row["dataset_index"]) for row in split_rows] != list(range(expected)):
            raise RuntimeError(f"{split} indices are incomplete or duplicated")
        annotated[split] = split_rows

    val_report = split_report(annotated["val"])
    test_report = split_report(annotated["test"])
    report = {
        "dataset": "HuggingFaceM4/ChartQA",
        "source_adapter": "Kxck/Finance_500_v1",
        "inference_engine": "vLLM merged BF16",
        "splits": {"val_0_500": val_report, "test_0_2500": test_report},
        "test_minus_val": {
            section: split_delta(val_report, test_report, section)
            for section in ("task_type", "answer_type", "operation_type")
        },
        "imbalance_summary": {
            "val_largest_task_by_volume": dominant(val_report, "task_type", "total"),
            "test_largest_task_by_volume": dominant(test_report, "task_type", "total"),
            "val_largest_failure_task": dominant(val_report, "task_type", "failures"),
            "test_largest_failure_task": dominant(test_report, "task_type", "failures"),
            "val_highest_task_error_rate": dominant(
                val_report, "task_type", "error_rate_pct"
            ),
            "test_highest_task_error_rate": dominant(
                test_report, "task_type", "error_rate_pct"
            ),
            "val_dominant_answer_type": dominant(val_report, "answer_type", "total"),
            "test_dominant_answer_type": dominant(test_report, "answer_type", "total"),
            "val_dominant_operation": dominant(val_report, "operation_type", "total"),
            "test_dominant_operation": dominant(test_report, "operation_type", "total"),
        },
        "taxonomy_precedence": [
            "yes/no target -> logical_reasoning",
            "explicit count intent -> counting",
            "arithmetic intent -> numerical_reasoning",
            "remaining extrema/color/category/value lookup -> visual_grounding",
        ],
        "labels_are_ground_truth": False,
        "teacher_has_been_used": False,
        "evaluation_only": True,
        "allowed_for_training": False,
    }

    for split, rows in annotated.items():
        output_path = output_dir / f"sft_{split}_distribution_labeled.jsonl"
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_path = output_dir / "sft_val500_test2500_distribution_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
