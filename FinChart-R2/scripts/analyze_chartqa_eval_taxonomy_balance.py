#!/usr/bin/env python3
"""Extract and profile the FinChart frozen ChartQA evaluation taxonomy.

The default scope matches the project evaluation boundary exactly:
  * ChartQA val[0:500]
  * ChartQA test[0:2500]

Labels are transparent deterministic heuristics derived from each question and
reference answer. They are useful for coverage and imbalance diagnostics, but
they are not manual or teacher-audited ground truth.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "dataset_balance"
DATASET_NAME = "HuggingFaceM4/ChartQA"
TAXONOMY_VERSION = "finchart_eval_heuristic_v1"

TASK_TYPES = (
    "numerical_reasoning",
    "counting",
    "visual_grounding",
    "logical_reasoning",
)
ANSWER_TYPES = ("yes_no", "numeric", "text")
OPERATION_TYPES = (
    "none",
    "lookup",
    "sum",
    "difference",
    "average",
    "median",
    "ratio",
    "percentage",
    "percentage_change",
    "count",
    "comparison",
    "max_difference",
    "min_max",
    "multi_step",
)

YES_NO = {"yes", "no", "true", "false"}
COLORS = {
    "black",
    "blue",
    "brown",
    "cyan",
    "gold",
    "gray",
    "green",
    "grey",
    "magenta",
    "orange",
    "pink",
    "purple",
    "red",
    "teal",
    "violet",
    "white",
    "yellow",
}

PATTERNS = {
    "percentage_change": re.compile(
        r"\b(?:percentage|percent)\s+(?:change|increase|decrease)\b|"
        r"\b(?:increase|decrease|changed?)\s+by\s+what\s+(?:percentage|percent)\b|"
        r"\bby\s+what\s+(?:percentage|percent)\b",
        re.I,
    ),
    "max_difference": re.compile(
        r"\b(?:maximum|largest|greatest|biggest|highest)\s+(?:absolute\s+)?difference\b",
        re.I,
    ),
    "average": re.compile(r"\b(?:average|arithmetic mean|mean value)\b", re.I),
    "median": re.compile(r"\bmedian\b", re.I),
    "sum": re.compile(
        r"\b(?:sum|add(?:ed)?|addition|plus|combined|total of)\b|"
        r"\badd\s+up\b|\btogether\b",
        re.I,
    ),
    "difference": re.compile(
        r"\b(?:difference|subtract(?:ed|ion)?|deduct(?:ed|ion)?|"
        r"how much (?:more|less)|more than|less than)\b",
        re.I,
    ),
    "ratio": re.compile(
        r"\b(?:ratio|divide(?:d)?|division|quotient|how many times|times as)\b",
        re.I,
    ),
    "percentage": re.compile(
        r"\b(?:what|which|calculate|find)\s+(?:is\s+the\s+)?(?:percentage|percent)\b|"
        r"\b(?:percentage|percent)\s+of\b|\bout of (?:a )?hundred\b",
        re.I,
    ),
    "count": re.compile(
        r"\b(?:how many|number of|count(?:ing)?|for how many|in how many)\b",
        re.I,
    ),
    "comparison": re.compile(
        r"\b(?:compare|comparison|greater than|less than|equal to|same as|"
        r"higher than|lower than|exceed(?:s|ed)?|at least|at most)\b",
        re.I,
    ),
    "min_max": re.compile(
        r"\b(?:minimum|maximum|min|max|smallest|largest|highest|lowest|"
        r"greatest|least|most)\b",
        re.I,
    ),
}

YES_NO_QUESTION = re.compile(
    r"^\s*(?:is|are|was|were|do|does|did|has|have|had|can|could|will|would)\b",
    re.I,
)
RATIO_COUNT_PHRASE = re.compile(r"\bhow many times\b", re.I)
SAMPLE_PROJECTION = re.compile(
    r"\b(?:ask|survey|sample)\b.*\bhow many\b|"
    r"\bhow many\b.*\b(?:people|respondents)\b.*\b(?:will|would)\b",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=PROJECT_ROOT.parent / ".hf_cache",
        help="Writable Hugging Face cache; no secret is read or written.",
    )
    return parser.parse_args()


def unwrap_answer(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def numeric_value(value: Any) -> float | None:
    text = normalized_text(value).replace(",", "")
    text = text.removesuffix("%").strip()
    try:
        return float(text)
    except ValueError:
        return None


def classify_answer(answer: Any, question: str) -> tuple[str, str]:
    text = normalized_text(answer)
    if text in YES_NO:
        return "yes_no", "boolean"
    value = numeric_value(answer)
    if value is not None:
        if "%" in str(answer):
            return "numeric", "percentage"
        if re.search(r"\byear\b", question, re.I) and value.is_integer() and 1000 <= value <= 2999:
            return "numeric", "year"
        return "numeric", "integer" if value.is_integer() else "decimal"
    if text in COLORS:
        return "text", "color"
    if isinstance(answer, (list, tuple)) and len(answer) > 1:
        return "text", "multiple_answers"
    return "text", "category_text"


def matched_signals(question: str) -> list[str]:
    return [name for name, pattern in PATTERNS.items() if pattern.search(question)]


def infer_operation(question: str, answer_type: str) -> tuple[str, list[str]]:
    signals = matched_signals(question)
    signal_set = set(signals)

    if "percentage_change" in signal_set:
        return "percentage_change", signals
    if "max_difference" in signal_set:
        return "max_difference", signals

    arithmetic = [
        name
        for name in ("average", "median", "sum", "difference", "ratio", "percentage")
        if name in signal_set
    ]
    if len(arithmetic) >= 2:
        return "multi_step", signals
    if arithmetic:
        return arithmetic[0], signals

    if (
        "count" in signal_set
        and not RATIO_COUNT_PHRASE.search(question)
        and not SAMPLE_PROJECTION.search(question)
    ):
        return "count", signals
    if answer_type == "yes_no" or YES_NO_QUESTION.search(question) or "comparison" in signal_set:
        return "comparison", signals
    if "min_max" in signal_set:
        return "min_max", signals
    return "lookup", signals


def infer_task_type(operation: str, answer_type: str) -> str:
    if answer_type == "yes_no" or operation == "comparison":
        return "logical_reasoning"
    if operation == "count":
        return "counting"
    if operation in {
        "sum",
        "difference",
        "average",
        "median",
        "ratio",
        "percentage",
        "percentage_change",
        "max_difference",
        "multi_step",
    }:
        return "numerical_reasoning"
    return "visual_grounding"


def classify_row(split: str, index: int, row: dict[str, Any]) -> dict[str, Any]:
    question = str(row["query"]).strip()
    raw_answer = row["label"]
    answer = unwrap_answer(raw_answer)
    answer_type, answer_subtype = classify_answer(answer, question)
    operation, signals = infer_operation(question, answer_type)
    task_type = infer_task_type(operation, answer_type)
    return {
        "dataset": DATASET_NAME,
        "split": split,
        "dataset_index": index,
        "question": question,
        "ground_truth": answer,
        "task_type": task_type,
        "answer_type": answer_type,
        "answer_subtype": answer_subtype,
        "operation_type": operation,
        "matched_signals": signals,
        "taxonomy_source": "deterministic_question_answer_heuristic",
        "taxonomy_version": TAXONOMY_VERSION,
        "taxonomy_is_ground_truth": False,
    }


def stream_metadata(dataset_name: str, split: str, limit: int) -> Iterable[dict[str, Any]]:
    from datasets import Image, load_dataset

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    if "image" in dataset.column_names:
        dataset = dataset.cast_column("image", Image(decode=False))
    yield from itertools.islice(dataset, limit)


def distribution(
    rows: list[dict[str, Any]], field: str, expected: tuple[str, ...] | None = None
) -> dict[str, dict[str, float | int]]:
    counts = Counter(str(row[field]) for row in rows)
    labels = list(expected or ()) + sorted(set(counts) - set(expected or ()))
    total = len(rows)
    return {
        label: {
            "count": counts.get(label, 0),
            "percentage": round(100.0 * counts.get(label, 0) / total, 2),
        }
        for label in labels
    }


def imbalance_diagnostic(
    values: dict[str, dict[str, float | int]], expected: tuple[str, ...]
) -> dict[str, Any]:
    counts = {label: int(values[label]["count"]) for label in expected}
    total = sum(counts.values())
    observed = [count for count in counts.values() if count > 0]
    dominant = max(counts, key=counts.get)
    minority = min(counts, key=counts.get)
    probabilities = [count / total for count in counts.values() if count]
    entropy = -sum(p * math.log(p) for p in probabilities)
    normalized_entropy = entropy / math.log(len(expected)) if len(expected) > 1 else 1.0
    zero_categories = [label for label, count in counts.items() if count == 0]
    ratio = round(max(observed) / min(observed), 2) if observed else None
    dominant_share = counts[dominant] / total if total else 0.0
    if zero_categories or dominant_share >= 0.65:
        severity = "high"
    elif dominant_share >= 0.50 or (ratio is not None and ratio >= 3.0):
        severity = "moderate"
    else:
        severity = "low"
    return {
        "severity": severity,
        "dominant_category": dominant,
        "dominant_percentage": round(100.0 * dominant_share, 2),
        "minority_category": minority,
        "minority_percentage": round(100.0 * counts[minority] / total, 2) if total else 0.0,
        "max_to_min_observed_ratio": ratio,
        "zero_count_categories": zero_categories,
        "normalized_entropy": round(normalized_entropy, 4),
    }


def split_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = {
        "task_type": TASK_TYPES,
        "answer_type": ANSWER_TYPES,
        "operation_type": OPERATION_TYPES,
    }
    answer_subtypes = distribution(rows, "answer_subtype")
    report: dict[str, Any] = {
        "records": len(rows),
        "distributions": {"answer_subtype": answer_subtypes},
        "imbalance": {},
    }
    for field, expected in dimensions.items():
        values = distribution(rows, field, expected)
        report["distributions"][field] = values
        report["imbalance"][field] = imbalance_diagnostic(values, expected)
    return report


def percentage_point_deltas(
    left: dict[str, Any], right: dict[str, Any], field: str
) -> dict[str, float]:
    left_values = left["distributions"][field]
    right_values = right["distributions"][field]
    labels = sorted(set(left_values) | set(right_values))
    return {
        label: round(
            float(right_values.get(label, {}).get("percentage", 0.0))
            - float(left_values.get(label, {}).get("percentage", 0.0)),
            2,
        )
        for label in labels
    }


def markdown_table(values: dict[str, dict[str, float | int]]) -> str:
    lines = ["| Category | Count | Share |", "|---|---:|---:|"]
    for label, stats in sorted(values.items(), key=lambda item: -int(item[1]["count"])):
        lines.append(f"| `{label}` | {stats['count']} | {stats['percentage']}% |")
    return "\n".join(lines)


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# ChartQA val[0:500] and test[0:2500] taxonomy balance",
        "",
        "> These are deterministic heuristic labels, not manual or teacher-audited ground truth.",
        "",
    ]
    for split in ("val_500", "test_2500", "combined_3000"):
        split_report_value = report["splits"][split]
        lines.extend([f"## {split}", ""])
        for field in ("task_type", "answer_type", "answer_subtype", "operation_type"):
            lines.extend(
                [
                    f"### {field}",
                    "",
                    markdown_table(split_report_value["distributions"][field]),
                    "",
                ]
            )
        lines.extend(["### Imbalance diagnostics", ""])
        for field, diagnostic in split_report_value["imbalance"].items():
            lines.append(
                f"- `{field}`: **{diagnostic['severity']}**, dominant "
                f"`{diagnostic['dominant_category']}` at "
                f"{diagnostic['dominant_percentage']}%."
            )
        lines.append("")
    lines.extend(
        [
            "## Test minus validation distribution shift",
            "",
            "Values are percentage-point changes from `val[0:500]` to `test[0:2500]`.",
            "",
            "```json",
            json.dumps(report["test_minus_val_percentage_points"], indent=2),
            "```",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.val_size <= 0 or args.test_size <= 0:
        raise ValueError("Slice sizes must be positive")

    cache = args.hf_cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache / "datasets"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, limit in (("val", args.val_size), ("test", args.test_size)):
        source_rows = list(stream_metadata(args.dataset, split, limit))
        if len(source_rows) != limit:
            raise RuntimeError(f"Expected {limit} {split} rows, received {len(source_rows)}")
        rows_by_split[split] = [
            classify_row(split, index, row) for index, row in enumerate(source_rows)
        ]
        print(f"Classified {len(rows_by_split[split])} rows from {split}[0:{limit}]")

    all_rows = rows_by_split["val"] + rows_by_split["test"]
    records_path = output_dir / "chartqa_val500_test2500_taxonomy.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    val_report = split_report(rows_by_split["val"])
    test_report = split_report(rows_by_split["test"])
    combined_report = split_report(all_rows)
    report = {
        "dataset": args.dataset,
        "scope": {"val": f"0:{args.val_size}", "test": f"0:{args.test_size}"},
        "records": len(all_rows),
        "taxonomy_version": TAXONOMY_VERSION,
        "taxonomy_source": "deterministic_question_answer_heuristic",
        "taxonomy_is_ground_truth": False,
        "closed_taxonomy": {
            "task_type": list(TASK_TYPES),
            "answer_type": list(ANSWER_TYPES),
            "operation_type": list(OPERATION_TYPES),
        },
        "splits": {
            f"val_{args.val_size}": val_report,
            f"test_{args.test_size}": test_report,
            f"combined_{len(all_rows)}": combined_report,
        },
        "test_minus_val_percentage_points": {
            field: percentage_point_deltas(val_report, test_report, field)
            for field in ("task_type", "answer_type", "answer_subtype", "operation_type")
        },
        "artifacts": {"records_jsonl": str(records_path)},
    }

    report_path = output_dir / "chartqa_val500_test2500_balance_report.json"
    markdown_path = output_dir / "chartqa_val500_test2500_balance_report.md"
    report["artifacts"].update(
        {"report_json": str(report_path), "report_markdown": str(markdown_path)}
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, markdown_path)
    print(json.dumps(report["splits"], ensure_ascii=False, indent=2))
    print(f"Wrote {records_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
