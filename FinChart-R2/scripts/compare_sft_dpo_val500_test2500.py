#!/usr/bin/env python3
"""Pair SFT and cumulative SFT+DPO predictions on val-500 and test-2500."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from analyze_sft_test2500_failures import (
    answer_type,
    deterministic_match,
    operation_heuristic,
    task_heuristic,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_DIR / "results" / "vali"
DEFAULT_SFT = (
    RESULTS
    / "sft_vllm_chartqa_3000"
    / "sft_vllm_chartqa_val500_test2500_predictions.jsonl"
)
DEFAULT_DPO = (
    RESULTS
    / "dpo_vllm_chartqa_3000"
    / "dpo_vllm_chartqa_val500_test2500_predictions.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results" / "comparison" / "sft_vs_dpo_val500_test2500"
)
EXPECTED = {"val": 500, "test": 2500}
STRUCTURED_FIELD = re.compile(
    r"(?im)^\s*(target series|target category|relevant values|operation|calculation|answer)\s*:"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--dpo", type=Path, default=DEFAULT_DPO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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


def exact_mcnemar_p(sft_only: int, dpo_only: int) -> float:
    discordant = sft_only + dpo_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(sft_only, dpo_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def group_metrics(
    rows: list[dict[str, Any]], label: Callable[[dict[str, Any]], str]
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(label(row), []).append(row)
    output = {}
    for name, subset in sorted(grouped.items()):
        total = len(subset)
        sft_correct = sum(bool(row["sft_correct"]) for row in subset)
        dpo_correct = sum(bool(row["dpo_correct"]) for row in subset)
        output[name] = {
            "total": total,
            "sft_correct": sft_correct,
            "sft_accuracy_pct": round(100 * sft_correct / total, 2),
            "dpo_correct": dpo_correct,
            "dpo_accuracy_pct": round(100 * dpo_correct / total, 2),
            "dpo_minus_sft_correct": dpo_correct - sft_correct,
            "dpo_minus_sft_accuracy_pp": round(
                100 * (dpo_correct - sft_correct) / total, 2
            ),
            "sft_correct_dpo_wrong": sum(
                row["transition"] == "SFT_CORRECT_DPO_WRONG" for row in subset
            ),
            "sft_wrong_dpo_correct": sum(
                row["transition"] == "SFT_WRONG_DPO_CORRECT" for row in subset
            ),
        }
    return output


def split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    transitions = Counter(row["transition"] for row in rows)
    sft_correct = sum(bool(row["sft_correct"]) for row in rows)
    dpo_correct = sum(bool(row["dpo_correct"]) for row in rows)
    sft_only = transitions["SFT_CORRECT_DPO_WRONG"]
    dpo_only = transitions["SFT_WRONG_DPO_CORRECT"]
    return {
        "total": total,
        "sft": {
            "correct": sft_correct,
            "incorrect": total - sft_correct,
            "accuracy": sft_correct / total,
        },
        "dpo": {
            "correct": dpo_correct,
            "incorrect": total - dpo_correct,
            "accuracy": dpo_correct / total,
        },
        "dpo_minus_sft": {
            "correct": dpo_correct - sft_correct,
            "accuracy_percentage_points": round(
                100 * (dpo_correct - sft_correct) / total, 4
            ),
        },
        "transitions": dict(transitions),
        "exact_mcnemar_p": exact_mcnemar_p(sft_only, dpo_only),
        "output_format": {
            "sft_structured_responses": sum(
                bool(STRUCTURED_FIELD.search(row["sft_prediction"])) for row in rows
            ),
            "dpo_structured_responses": sum(
                bool(STRUCTURED_FIELD.search(row["dpo_prediction"])) for row in rows
            ),
            "sft_length_finishes": sum(
                row.get("sft_finish_reason") == "length" for row in rows
            ),
            "dpo_length_finishes": sum(
                row.get("dpo_finish_reason") == "length" for row in rows
            ),
        },
        "by_task_type_heuristic": group_metrics(
            rows, lambda row: row["task_type_heuristic"]
        ),
        "by_answer_type": group_metrics(rows, lambda row: row["answer_type"]),
        "by_operation_type_heuristic": group_metrics(
            rows, lambda row: row["operation_type_heuristic"]
        ),
    }


def main() -> None:
    args = parse_args()
    sft_rows = read_jsonl(args.sft.resolve())
    dpo_rows = read_jsonl(args.dpo.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paired_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, expected in EXPECTED.items():
        sft = {
            int(row["dataset_index"]): row
            for row in sft_rows
            if row.get("split") == split
        }
        dpo = {
            int(row["dataset_index"]): row
            for row in dpo_rows
            if row.get("split") == split
        }
        if set(sft) != set(range(expected)) or set(dpo) != set(range(expected)):
            raise RuntimeError(f"Incomplete or duplicate {split} indices")
        paired = []
        for index in range(expected):
            left, right = sft[index], dpo[index]
            if left["question"] != right["question"]:
                raise RuntimeError(f"Question mismatch at {split}[{index}]")
            if left["ground_truth"] != right["ground_truth"]:
                raise RuntimeError(f"Ground-truth mismatch at {split}[{index}]")
            sft_correct = deterministic_match(left["prediction"], left["ground_truth"])
            dpo_correct = deterministic_match(right["prediction"], right["ground_truth"])
            if sft_correct != bool(left["deterministic_correct"]):
                raise RuntimeError(f"SFT matcher mismatch at {split}[{index}]")
            if dpo_correct != bool(right["deterministic_correct"]):
                raise RuntimeError(f"DPO matcher mismatch at {split}[{index}]")
            transition = (
                "BOTH_CORRECT"
                if sft_correct and dpo_correct
                else "SFT_CORRECT_DPO_WRONG"
                if sft_correct
                else "SFT_WRONG_DPO_CORRECT"
                if dpo_correct
                else "BOTH_WRONG"
            )
            operation = operation_heuristic(left["question"], left["ground_truth"])
            paired.append(
                {
                    "dataset": "HuggingFaceM4/ChartQA",
                    "split": split,
                    "dataset_index": index,
                    "question": left["question"],
                    "ground_truth": left["ground_truth"],
                    "sft_prediction": left["prediction"],
                    "dpo_prediction": right["prediction"],
                    "sft_correct": sft_correct,
                    "dpo_correct": dpo_correct,
                    "transition": transition,
                    "task_type_heuristic": task_heuristic(operation),
                    "answer_type": answer_type(left["ground_truth"]),
                    "operation_type_heuristic": operation,
                    "sft_finish_reason": left.get("finish_reason"),
                    "dpo_finish_reason": right.get("finish_reason"),
                    "labels_are_ground_truth": False,
                    "evaluation_only": True,
                    "allowed_for_training": False,
                }
            )
        paired_by_split[split] = paired
        write_jsonl(output_dir / f"sft_vs_dpo_{split}_paired.jsonl", paired)

    combined = paired_by_split["val"] + paired_by_split["test"]
    report = {
        "dataset": "HuggingFaceM4/ChartQA",
        "sft_adapter": "Kxck/Finance_500_v1",
        "dpo_adapter": "Kxck/Finance_500_v1_DPO_386_provisional",
        "engine": "vLLM 0.28 OpenAI-compatible server; merged BF16",
        "protocol_matched": True,
        "splits": {
            "val_0_500": split_summary(paired_by_split["val"]),
            "test_0_2500": split_summary(paired_by_split["test"]),
        },
        "combined_descriptive_only": split_summary(combined),
        "labels_are_ground_truth": False,
        "evaluation_only": True,
        "allowed_for_training": False,
        "guardrail": "Validation and test comparisons must never enter training.",
    }
    report_path = output_dir / "sft_vs_dpo_val500_test2500_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
