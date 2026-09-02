#!/usr/bin/env python3
"""Analyze SFT failures on ChartQA test[0:2500] before teacher labeling.

The task and operation labels produced here are transparent deterministic
heuristics, not teacher labels or ground truth.  The output remains an
evaluation/diagnosis artifact: ChartQA test examples must never be used for
continued SFT, DPO, or any other model optimization.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_DIR / ".cache" / "datasets"))

from datasets import load_dataset


DATASET_NAME = "HuggingFaceM4/ChartQA"
INPUT_DEFAULT = (
    PROJECT_DIR
    / "results"
    / "vali"
    / "sft_vllm_chartqa_3000"
    / "sft_vllm_chartqa_val500_test2500_predictions.jsonl"
)
OUTPUT_DEFAULT = PROJECT_DIR / "results" / "phase2d_sft_failures_test_2500"

COUNT_INTENT = re.compile(
    r"\b(how many|number of|for how many|in how many|count)\b", re.I
)
COUNT_ARITHMETIC_EXCEPTION = re.compile(
    r"\b(how many (?:more|fewer|less|times|percent)|number of times)\b", re.I
)
AVERAGE_INTENT = re.compile(r"\b(average|mean)\b", re.I)
MEDIAN_INTENT = re.compile(r"\bmedian\b", re.I)
RATIO_INTENT = re.compile(r"\b(ratio|divide|divided|division|how many times)\b", re.I)
PRODUCT_INTENT = re.compile(r"\b(product|multiply|multiplication)\b", re.I)
DIFFERENCE_INTENT = re.compile(
    r"\b(difference|subtract|deduct|how many more|how many fewer|"
    r"how much more|how much less|gap)\b",
    re.I,
)
PERCENTAGE_CHANGE_INTENT = re.compile(
    r"\b(percent(?:age)? change|percent(?:age)? (?:increase|decrease)|"
    r"increase by what percent|decrease by what percent)\b",
    re.I,
)
PERCENTAGE_INTENT = re.compile(r"\b(percent|percentage)\b", re.I)
SUM_INTENT = re.compile(
    r"\b(sum|add(?:ed|ition| up)?|combined|together)\b|"
    r"\btotal of\b|\btotal\s+(?:value|percentage|market share)\s+of\b.*\b(?:and|between|through)\b",
    re.I,
)
EXTREMA_INTENT = re.compile(
    r"\b(maximum|minimum|highest|lowest|largest|smallest|peak|most|least)\b",
    re.I,
)
COLOR_INTENT = re.compile(r"\b(colou?r|line represents|which line|legend)\b", re.I)
STRUCTURED_FIELD = re.compile(
    r"(?im)^\s*(target series|target category|relevant values|operation|calculation|answer)\s*:"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow a ChartQA download when it is not already cached.",
    )
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def unwrap_answer(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def normalize_answer(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.lower().strip()
    text = re.sub(r"^\s*(?:final\s+)?answer\s*:\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*/\s*", "/", text)
    return text.strip(string.whitespace + ".,;:!?")


def as_number(value: Any) -> float | None:
    try:
        return float(normalize_answer(value).replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return None


def deterministic_match(left: Any, right: Any) -> bool:
    left_text, right_text = normalize_answer(left), normalize_answer(right)
    if left_text == right_text:
        return True
    left_number, right_number = as_number(left_text), as_number(right_text)
    return (
        left_number is not None
        and right_number is not None
        and abs(left_number - right_number) <= 1e-6
    )


def answer_type(value: Any) -> str:
    text = normalize_answer(value)
    if text in {"yes", "no"}:
        return "yes_no"
    return "numeric" if as_number(text) is not None else "text"


def extract_embedded_answer(prediction: Any) -> str:
    text = str(prediction or "").strip()
    answer_lines = list(
        re.finditer(r"(?im)^\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$", text)
    )
    if answer_lines:
        return answer_lines[-1].group(1).strip()
    answer_sentence = re.search(
        r"(?i)\b(?:the\s+)?answer\s+is\s+([^\n.]+)", text
    )
    if answer_sentence:
        return answer_sentence.group(1).strip()
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    last_line = nonempty_lines[-1] if nonempty_lines else text
    if "=" in last_line:
        return last_line.rsplit("=", 1)[-1].strip()
    return text


def representation_scale_100(left: Any, right: Any) -> bool:
    left_number, right_number = as_number(left), as_number(right)
    if left_number is None or right_number is None:
        return False
    return (
        abs(left_number - 100 * right_number) <= 1e-6
        or abs(100 * left_number - right_number) <= 1e-6
    )


def operation_heuristic(question: str, ground_truth: Any) -> str:
    if answer_type(ground_truth) == "yes_no":
        return "comparison"
    if COUNT_INTENT.search(question) and not COUNT_ARITHMETIC_EXCEPTION.search(question):
        return "count"
    for name, pattern in (
        ("average", AVERAGE_INTENT),
        ("median", MEDIAN_INTENT),
        ("ratio", RATIO_INTENT),
        ("product", PRODUCT_INTENT),
        ("difference", DIFFERENCE_INTENT),
        ("percentage_change", PERCENTAGE_CHANGE_INTENT),
        ("percentage", PERCENTAGE_INTENT),
        ("sum", SUM_INTENT),
        ("extrema", EXTREMA_INTENT),
        ("color_lookup", COLOR_INTENT),
    ):
        if pattern.search(question):
            return name
    return "lookup"


def task_heuristic(operation: str) -> str:
    if operation == "comparison":
        return "logical_reasoning"
    if operation == "count":
        return "counting"
    if operation in {
        "average",
        "median",
        "ratio",
        "product",
        "difference",
        "percentage_change",
        "percentage",
        "sum",
    }:
        return "numerical_reasoning"
    return "visual_grounding"


def distribution(counter: Counter[str], total: int) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "count": count,
            "percentage": round(100 * count / total, 2) if total else 0.0,
        }
        for name, count in counter.most_common()
    }


def main() -> None:
    args = parse_args()
    source_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_jsonl(source_path)
    rows = [row for row in source_rows if row.get("split") == "test"]
    if len(rows) != 2500:
        raise RuntimeError(f"Expected 2,500 test rows, found {len(rows)}")
    rows.sort(key=lambda row: int(row["dataset_index"]))
    indices = [int(row["dataset_index"]) for row in rows]
    if indices != list(range(2500)):
        raise RuntimeError("Test rows must contain every index from 0 through 2499 once")

    if not args.allow_network:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    dataset = load_dataset(DATASET_NAME, split="test")
    if len(dataset) != 2500:
        raise RuntimeError(f"Expected ChartQA test size 2,500, found {len(dataset)}")

    analyzed: list[dict[str, Any]] = []
    for row in rows:
        index = int(row["dataset_index"])
        example = dataset[index]
        question = str(example.get("query", example.get("question", ""))).strip()
        ground_truth = str(
            unwrap_answer(example.get("label", example.get("answer", "")))
        ).strip()
        if question != str(row["question"]).strip():
            raise RuntimeError(f"Question mismatch at test[{index}]")
        if ground_truth != str(row["ground_truth"]).strip():
            raise RuntimeError(f"Ground-truth mismatch at test[{index}]")
        locally_correct = deterministic_match(row["prediction"], ground_truth)
        if locally_correct != bool(row["deterministic_correct"]):
            raise RuntimeError(f"Matcher mismatch at test[{index}]")

        operation = operation_heuristic(question, ground_truth)
        task = task_heuristic(operation)
        embedded_answer = extract_embedded_answer(row["prediction"])
        format_recoverable = not locally_correct and deterministic_match(
            embedded_answer, ground_truth
        )
        representation_candidate = (
            not locally_correct
            and not format_recoverable
            and representation_scale_100(embedded_answer, ground_truth)
        )
        signals: list[str] = []
        if format_recoverable:
            signals.append("FORMAT_RECOVERABLE")
        if representation_candidate:
            signals.append("PERCENTAGE_PROPORTION_EQUIVALENCE_CANDIDATE")
        if row.get("finish_reason") == "length":
            signals.append("MAX_TOKEN_TRUNCATION")
        if STRUCTURED_FIELD.search(str(row.get("prediction", ""))):
            signals.append("STRUCTURED_OUTPUT_DRIFT")
        if not locally_correct and not signals:
            signals.append("SUBSTANTIVE_MISMATCH")

        source_label = int(example["human_or_machine"])
        exported = dict(row)
        exported.update(
            {
                "record_version": "phase2d_sft_test2500_failure_analysis_v1",
                "image_split": "test",
                "image_index": index,
                "question_source": "human" if source_label == 0 else "machine",
                "answer_type": answer_type(ground_truth),
                "task_type_heuristic": task,
                "operation_heuristic": operation,
                "embedded_answer_candidate": embedded_answer,
                "format_recoverable": format_recoverable,
                "representation_equivalence_candidate": representation_candidate,
                "failure_signals": signals,
                "teacher_priority_candidate": bool(
                    not locally_correct
                    and not format_recoverable
                    and not representation_candidate
                ),
                "labels_are_ground_truth": False,
                "evaluation_only": True,
                "allowed_for_training": False,
            }
        )
        analyzed.append(exported)

    failures = [row for row in analyzed if not row["deterministic_correct"]]
    recoverable = [row for row in failures if row["format_recoverable"]]
    representation = [
        row for row in failures if row["representation_equivalence_candidate"]
    ]
    teacher_priority = [row for row in failures if row["teacher_priority_candidate"]]

    all_task = Counter(row["task_type_heuristic"] for row in analyzed)
    failure_task = Counter(row["task_type_heuristic"] for row in failures)
    priority_task = Counter(row["task_type_heuristic"] for row in teacher_priority)
    operation_counts = Counter(row["operation_heuristic"] for row in failures)
    signal_counts = Counter(
        signal for row in failures for signal in row["failure_signals"]
    )
    source_totals = Counter(row["question_source"] for row in analyzed)
    source_failures = Counter(row["question_source"] for row in failures)

    task_error_rates = {
        task: {
            "total": all_task[task],
            "strict_failures": failure_task[task],
            "strict_error_rate": round(100 * failure_task[task] / all_task[task], 2),
            "teacher_priority_after_local_gates": priority_task[task],
        }
        for task in sorted(all_task)
    }
    source_error_rates = {
        source: {
            "total": source_totals[source],
            "strict_failures": source_failures[source],
            "strict_error_rate": round(
                100 * source_failures[source] / source_totals[source], 2
            ),
        }
        for source in ("human", "machine")
    }

    full_path = output_dir / "sft_test_2500_predictions_analyzed.jsonl"
    failures_path = output_dir / "sft_test_2500_failures_676.jsonl"
    failures_json_path = output_dir / "sft_test_2500_failures_676.json"
    priority_path = output_dir / "sft_test_2500_teacher_priority_after_local_gates.jsonl"
    report_path = output_dir / "sft_test_2500_failure_report.json"
    write_jsonl(full_path, analyzed)
    write_jsonl(failures_path, failures)
    write_jsonl(priority_path, teacher_priority)
    failures_json_path.write_text(
        json.dumps(failures, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = {
        "dataset": DATASET_NAME,
        "split": "test",
        "range": "0:2500",
        "source_adapter": "Kxck/Finance_500_v1",
        "inference_engine": "vLLM merged BF16",
        "total": len(analyzed),
        "strict_correct": len(analyzed) - len(failures),
        "strict_failures": len(failures),
        "strict_accuracy": round((len(analyzed) - len(failures)) / len(analyzed), 6),
        "strict_error_rate": round(len(failures) / len(analyzed), 6),
        "local_failure_gates": {
            "format_recoverable": len(recoverable),
            "percentage_proportion_equivalence_candidates": len(representation),
            "remaining_teacher_priority": len(teacher_priority),
        },
        "strict_failure_task_distribution_heuristic": distribution(
            failure_task, len(failures)
        ),
        "teacher_priority_task_distribution_heuristic": distribution(
            priority_task, len(teacher_priority)
        ),
        "task_error_rates_heuristic": task_error_rates,
        "failure_operation_distribution_heuristic": distribution(
            operation_counts, len(failures)
        ),
        "failure_signal_counts_overlapping": dict(signal_counts.most_common()),
        "question_source_error_rates": source_error_rates,
        "dominant_strict_failure_task": failure_task.most_common(1)[0][0],
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
        "guardrail": "ChartQA test labels and teacher annotations must not enter training.",
        "outputs": {
            "all_analyzed_jsonl": full_path.name,
            "failures_jsonl": failures_path.name,
            "failures_json": failures_json_path.name,
            "teacher_priority_after_local_gates_jsonl": priority_path.name,
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
