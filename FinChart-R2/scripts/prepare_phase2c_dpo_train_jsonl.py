"""Canonicalize a Phase 2C train-mining run before teacher auditing.

This is a local-only transformation: it never loads a model, chart image, or
credential.  It preserves each raw SFT prediction and produces two stable
train-only JSONL files:

* dpo_train_correct.jsonl -- records already accepted by the original matcher.
* dpo_train_error.jsonl -- candidates for final-answer recovery and teacher audit.

The files are *not* DPO preference pairs.  In particular, correct records do
not have a rejected answer, so they cannot be passed to DPOTrainer directly.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "results" / "finchart_r2_phase2c_train_mining"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--run-tag", default="train_500_2500")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to --input-dir; use another directory to leave the source folder unchanged.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


ANSWER_LINE = re.compile(r"^\s*(?:final\s+answer|answer)\s*:\s*(.+?)\s*$", re.I)


def extract_final_answer(prediction: Any) -> str | None:
    """Return the last explicit Answer: field, without judging correctness."""
    text = "" if prediction is None else str(prediction)
    matches = [ANSWER_LINE.match(line) for line in text.splitlines()]
    extracted = [match.group(1).strip() for match in matches if match]
    return extracted[-1] if extracted else None


def canonical_record(row: dict[str, Any], source_file: str, route: str) -> dict[str, Any]:
    required = {
        "dataset_index",
        "image_dataset",
        "image_split",
        "image_index",
        "question",
        "ground_truth",
        "sft_prediction",
        "deterministic_correct",
        "candidate_status",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"{source_file} is missing required keys: {missing}")
    raw_prediction = str(row["sft_prediction"])
    return {
        "record_version": "phase2c_dpo_train_v1",
        "dataset_index": int(row["dataset_index"]),
        "image_dataset": str(row["image_dataset"]),
        "image_split": str(row["image_split"]),
        "image_index": int(row["image_index"]),
        "question": str(row["question"]),
        "ground_truth": row["ground_truth"],
        "ground_truth_normalized": row.get("ground_truth_normalized"),
        "sft_prediction_raw": raw_prediction,
        "original_prediction_normalized": row.get("prediction_normalized"),
        "extracted_final_answer": extract_final_answer(raw_prediction),
        "original_deterministic_correct": bool(row["deterministic_correct"]),
        "original_candidate_status": str(row["candidate_status"]),
        "audit_route": route,
        "source_file": source_file,
        "source_adapter": row.get("source_adapter"),
        "inference_backend": row.get("inference_backend"),
        "prompt_protocol": row.get("prompt_protocol"),
    }


def assert_disjoint(correct: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    correct_ids = [row["dataset_index"] for row in correct]
    error_ids = [row["dataset_index"] for row in errors]
    if len(correct_ids) != len(set(correct_ids)):
        raise ValueError("Duplicate dataset_index found in correct source JSONL")
    if len(error_ids) != len(set(error_ids)):
        raise ValueError("Duplicate dataset_index found in error source JSONL")
    overlap = sorted(set(correct_ids) & set(error_ids))
    if overlap:
        raise ValueError(f"Source JSONLs overlap, first dataset indices: {overlap[:10]}")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    correct_name = f"phase2c_{args.run_tag}_sft_correct.jsonl"
    error_name = f"phase2c_{args.run_tag}_sft_errors.jsonl"

    correct_source = read_jsonl(input_dir / correct_name)
    error_source = read_jsonl(input_dir / error_name)
    assert_disjoint(correct_source, error_source)

    correct = sorted(
        (canonical_record(row, correct_name, "RETAINED_CORRECT") for row in correct_source),
        key=lambda row: row["dataset_index"],
    )
    errors = sorted(
        (canonical_record(row, error_name, "RECHECK_THEN_TEACHER_AUDIT") for row in error_source),
        key=lambda row: row["dataset_index"],
    )

    # Keep this explicit: these files have no chosen/rejected preference fields.
    for row in correct + errors:
        row["is_dpo_preference_pair"] = False

    correct_output = output_dir / "dpo_train_correct.jsonl"
    error_output = output_dir / "dpo_train_error.jsonl"
    manifest_output = output_dir / "dpo_train_prepare_manifest.json"
    write_jsonl(correct_output, correct)
    write_jsonl(error_output, errors)
    manifest = {
        "record_version": "phase2c_dpo_train_v1",
        "source_run_tag": args.run_tag,
        "source_dir": str(input_dir),
        "correct_records": len(correct),
        "error_candidates": len(errors),
        "total_records": len(correct) + len(errors),
        "output_correct": str(correct_output),
        "output_error": str(error_output),
        "next_step": "Recheck dpo_train_error.jsonl with extracted_final_answer, then teacher-audit only unresolved candidates.",
        "dpo_pairs_created": False,
    }
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
