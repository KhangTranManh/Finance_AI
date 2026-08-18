"""Compare FinChart Phase 1 base predictions with Phase 2B SFT predictions.

The script writes local-only JSONL/JSON artifacts beneath FinChart-R2/results/comparison.
It has two complementary views:

1. Deterministic transitions across every shared validation sample:
   BOTH_CORRECT, BASE_WRONG_SFT_CORRECT, BASE_CORRECT_SFT_WRONG, BOTH_WRONG.
2. Phase 1 confirmed errors only. Their error type is the semantic-audited
   Phase 1 judge taxonomy, so the SFT_FIXED/STILL_WRONG counts are trustworthy.

For regressions, Phase 1 has no error label because the base answer was correct.
The script therefore assigns an explicitly heuristic question_type; it must not
be reported as a semantic error diagnosis without manual or judge review.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DETERMINISTIC = (
    ROOT / "FinChart-R1/results/qwen3vl4b_chartqa_val_0_500_deterministic.csv"
)
DEFAULT_BASE_FINAL = (
    ROOT / "FinChart-R1/results/qwen3vl4b_chartqa_val_0_500_phase1_final.csv"
)
DEFAULT_SFT = ROOT / "FinChart-R2/results/vali/sft_val_500.csv"
DEFAULT_OUTPUT_DIR = ROOT / "FinChart-R2/results/comparison"

CONFIRMED_ERROR_TYPES = {
    "NUMERICAL_REASONING",
    "VISUAL_EXTRACTION",
    "COUNTING",
    "LOGICAL_REASONING",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-deterministic", type=Path, default=DEFAULT_BASE_DETERMINISTIC)
    parser.add_argument("--base-final", type=Path, default=DEFAULT_BASE_FINAL)
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    """Parse CSV booleans without relying on pandas' inferred dtype."""
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}
    ).fillna(False).astype(bool)


def infer_question_type(question: str) -> str:
    """Transparent lexical fallback used only where no Phase 1 judge type exists."""
    text = str(question).lower()
    if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b|\btimes\b|\bpoints?\b", text):
        return "COUNTING"
    if re.search(
        r"\bsum\b|\btotal\b|\bdifference\b|\bdeduct\b|\baverage\b|\bmedian\b|"
        r"\bratio\b|\bpercentage\b|\bpercent\b|\bincrease\b|\bdecrease\b|\bmore than\b|"
        r"\bless than\b|\bhighest\b|\blowest\b|\bmax(?:imum)?\b|\bmin(?:imum)?\b",
        text,
    ):
        return "NUMERICAL_REASONING"
    if re.search(r"^is\b|^does\b|^which\b|\btrue\b|\bfalse\b|\bcompare\b|\bremains\b", text):
        return "LOGICAL_REASONING"
    return "VISUAL_EXTRACTION"


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def count_by_type(frame: pd.DataFrame, row_mask: pd.Series) -> dict[str, int]:
    return {
        error_type: int((row_mask & frame["question_type"].eq(error_type)).sum())
        for error_type in sorted(CONFIRMED_ERROR_TYPES)
    }


def main() -> None:
    args = parse_args()
    for path in (args.base_deterministic, args.base_final, args.sft):
        if not path.is_file():
            raise FileNotFoundError(path)

    base_det = pd.read_csv(args.base_deterministic)
    base_final = pd.read_csv(args.base_final)
    sft = pd.read_csv(args.sft)

    require_columns(
        base_det,
        {"dataset_index", "question", "ground_truth", "prediction", "deterministic_correct"},
        "base deterministic CSV",
    )
    require_columns(
        base_final,
        {"dataset_index", "final_verdict", "judge_error_type"},
        "base final CSV",
    )
    require_columns(sft, {"idx", "question", "gt", "pred", "correct"}, "SFT CSV")

    frame = (
        base_det[
            ["dataset_index", "question", "ground_truth", "prediction", "deterministic_correct"]
        ]
        .merge(
            base_final[["dataset_index", "final_verdict", "judge_error_type"]],
            on="dataset_index",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            sft[["idx", "question", "gt", "pred", "correct"]],
            left_on="dataset_index",
            right_on="idx",
            how="inner",
            validate="one_to_one",
            suffixes=("_base", "_sft"),
        )
    )

    if len(frame) != len(base_det) or len(frame) != len(sft):
        raise ValueError(
            f"Incomplete merge: base={len(base_det)}, sft={len(sft)}, merged={len(frame)}"
        )
    if not frame["question_base"].eq(frame["question_sft"]).all():
        raise ValueError("Question text differs between base and SFT files; refuse to compare.")

    frame["base_deterministic_correct"] = as_bool(frame["deterministic_correct"])
    frame["sft_deterministic_correct"] = as_bool(frame["correct"])
    base_ok = frame["base_deterministic_correct"]
    sft_ok = frame["sft_deterministic_correct"]

    frame["transition_tag"] = "BOTH_WRONG"
    frame.loc[base_ok & sft_ok, "transition_tag"] = "BOTH_CORRECT"
    frame.loc[~base_ok & sft_ok, "transition_tag"] = "BASE_WRONG_SFT_CORRECT"
    frame.loc[base_ok & ~sft_ok, "transition_tag"] = "BASE_CORRECT_SFT_WRONG"

    frame["phase1_confirmed_error"] = frame["final_verdict"].eq("INCORRECT")
    frame["phase1_error_type"] = frame["judge_error_type"].where(
        frame["judge_error_type"].isin(CONFIRMED_ERROR_TYPES),
        pd.NA,
    )
    frame["question_type"] = frame["phase1_error_type"]
    frame["question_type_source"] = "PHASE1_JUDGE"
    unresolved_type = frame["question_type"].isna()
    frame.loc[unresolved_type, "question_type"] = frame.loc[
        unresolved_type, "question_base"
    ].map(infer_question_type)
    frame.loc[unresolved_type, "question_type_source"] = "HEURISTIC"

    frame["confirmed_error_outcome"] = pd.NA
    confirmed = frame["phase1_confirmed_error"]
    frame.loc[confirmed & sft_ok, "confirmed_error_outcome"] = "SFT_FIXED"
    frame.loc[confirmed & ~sft_ok, "confirmed_error_outcome"] = "SFT_STILL_WRONG"

    transition_order = [
        "BOTH_CORRECT",
        "BASE_WRONG_SFT_CORRECT",
        "BASE_CORRECT_SFT_WRONG",
        "BOTH_WRONG",
    ]
    transitions = {
        tag: int(frame["transition_tag"].eq(tag).sum()) for tag in transition_order
    }

    confirmed_summary: list[dict[str, Any]] = []
    for error_type in sorted(CONFIRMED_ERROR_TYPES):
        rows = frame[confirmed & frame["question_type"].eq(error_type)]
        confirmed_summary.append(
            {
                "error_type": error_type,
                "base_errors": int(len(rows)),
                "sft_fixed": int(rows["confirmed_error_outcome"].eq("SFT_FIXED").sum()),
                "sft_still_wrong": int(
                    rows["confirmed_error_outcome"].eq("SFT_STILL_WRONG").sum()
                ),
            }
        )

    regression_mask = frame["transition_tag"].eq("BASE_CORRECT_SFT_WRONG")
    summary = {
        "samples": int(len(frame)),
        "transition_counts": transitions,
        "confirmed_phase1_errors": confirmed_summary,
        "regressions_by_question_type": count_by_type(frame, regression_mask),
        "notes": [
            "transition_tag uses the common deterministic matcher outputs from both CSV files.",
            "confirmed_phase1_errors uses only final_verdict=INCORRECT and the Phase 1 semantic judge error type.",
            "regressions_by_question_type is heuristic where no Phase 1 judge error type exists.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "phase1_vs_sft_500_comparison.jsonl"
    summary_path = args.output_dir / "phase1_vs_sft_500_summary.json"
    comparison_rows = frame[
        [
            "dataset_index",
            "question_base",
            "ground_truth",
            "prediction",
            "pred",
            "base_deterministic_correct",
            "sft_deterministic_correct",
            "transition_tag",
            "final_verdict",
            "phase1_error_type",
            "question_type",
            "question_type_source",
            "confirmed_error_outcome",
        ]
    ].rename(
        columns={
            "question_base": "question",
            "prediction": "base_prediction",
            "pred": "sft_prediction",
        }
    ).to_dict("records")
    with comparison_path.open("w", encoding="utf-8") as handle:
        for row in comparison_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved comparison: {comparison_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
