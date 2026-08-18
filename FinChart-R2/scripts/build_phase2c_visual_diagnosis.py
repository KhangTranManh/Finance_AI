"""Build the Phase 2C visual-grounding review queue from Base and SFT results.

This script deliberately separates authoritative labels from triage proposals:
  - Phase 1 confirmed error type comes from its semantic-audited final CSV.
  - proposed_subtype is a transparent question/prediction heuristic only.
  - final_subtype remains empty until manual or teacher review.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_DETERMINISTIC = ROOT / "FinChart-R1/results/qwen3vl4b_chartqa_val_0_500_deterministic.csv"
BASE_FINAL = ROOT / "FinChart-R1/results/qwen3vl4b_chartqa_val_0_500_phase1_final.csv"
SFT_RESULTS = ROOT / "FinChart-R2/results/vali/sft_val_500.csv"
OUTPUT_DIR = ROOT / "FinChart-R2/results/phase2c_visual_diagnosis"

VISUAL_TYPES = {"VISUAL_EXTRACTION", "COUNTING"}
SUBTYPES = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-deterministic", type=Path, default=BASE_DETERMINISTIC)
    parser.add_argument("--base-final", type=Path, default=BASE_FINAL)
    parser.add_argument("--sft", type=Path, default=SFT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}
    ).fillna(False).astype(bool)


def coarse_question_type(question: str) -> str:
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


def propose_visual_subtype(question: str, base_prediction: str, sft_prediction: str) -> tuple[str, str]:
    """Return a review-priority proposal, never a final diagnosis."""
    text = f"{question} {base_prediction} {sft_prediction}".lower()
    question_text = str(question).lower()

    if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b|\btimes\b|\bintercept", question_text):
        return "COUNTING_ERROR", "count_or_intersection"
    if re.search(r"\blegend\b|\brepresent(?:s|ed)?\b|\bwhich (?:line|bar|graph)\b", question_text):
        return "LEGEND_ASSOCIATION", "legend_or_label_mapping"
    if re.search(r"\bcolor\b|\bblue\b|\bred\b|\bgreen\b|\byellow\b|\bgrey\b|\bgray\b|\bbrown\b", question_text):
        return "WRONG_COLOR", "color_encoding"
    if re.search(r"\bhighest\b|\blowest\b|\bmaximum\b|\bminimum\b|\bpeak\b|\brightmost\b|\bleftmost\b", question_text):
        return "EXTREMA_ERROR", "extrema_or_position"
    if re.search(r"\baxis\b|\bx-axis\b|\by-axis\b|\bvertical\b|\bhorizontal\b", question_text):
        return "AXIS_ALIGNMENT", "axis_or_coordinate"
    if re.search(r"\bpoint\b|\bintersection\b|\bcross(?:ing)?\b", question_text):
        return "WRONG_POINT", "plotted_point"
    if re.search(r"\byear\b|\bmonth\b|\bcategory\b|\bwhich (?:year|country|group)\b", question_text):
        return "WRONG_CATEGORY", "category_or_time"
    if re.search(r"\bsmall\b|\btiny\b|\btext\b|\blabel\b|\btitle\b|\bocr\b", text):
        return "CROP_SMALL_TEXT", "small_text_or_ocr"
    if re.search(r"\bvalue\b|\bpercent\b|\bpercentage\b|\bwhat is\b", question_text):
        return "WRONG_VALUE", "value_reading"
    return "OTHER_VISUAL", "manual_localization_required"


def main() -> None:
    args = parse_args()
    for path in (args.base_deterministic, args.base_final, args.sft):
        if not path.is_file():
            raise FileNotFoundError(path)

    base = pd.read_csv(args.base_deterministic)
    final = pd.read_csv(args.base_final)
    sft = pd.read_csv(args.sft)
    frame = (
        base[
            ["dataset_index", "question", "ground_truth", "prediction", "deterministic_correct"]
        ]
        .merge(
            final[["dataset_index", "final_verdict", "judge_error_type", "judge_reason"]],
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
    if len(frame) != 500 or not frame["question_base"].eq(frame["question_sft"]).all():
        raise ValueError("Expected an exact 500-row, question-aligned comparison.")

    base_ok = as_bool(frame["deterministic_correct"])
    sft_ok = as_bool(frame["correct"])
    frame["transition_tag"] = "OTHER"
    frame.loc[~base_ok & ~sft_ok, "transition_tag"] = "BOTH_WRONG"
    frame.loc[base_ok & ~sft_ok, "transition_tag"] = "BASE_CORRECT_SFT_WRONG"

    frame["coarse_question_type"] = frame["question_base"].map(coarse_question_type)
    base_confirmed_visual = (
        frame["final_verdict"].eq("INCORRECT")
        & frame["judge_error_type"].isin(VISUAL_TYPES)
    )
    heuristic_visual_or_count = frame["coarse_question_type"].isin(VISUAL_TYPES)
    priority_transition = frame["transition_tag"].isin(
        ["BOTH_WRONG", "BASE_CORRECT_SFT_WRONG"]
    )

    candidates = frame[priority_transition & (base_confirmed_visual | heuristic_visual_or_count)].copy()
    candidates["candidate_source"] = "HEURISTIC_VISUAL_OR_COUNTING"
    candidates.loc[base_confirmed_visual.loc[candidates.index], "candidate_source"] = (
        "PHASE1_CONFIRMED_VISUAL_OR_COUNTING"
    )
    proposals = candidates.apply(
        lambda row: propose_visual_subtype(
            row["question_base"], row["prediction"], row["pred"]
        ),
        axis=1,
    )
    candidates[["proposed_subtype", "proposed_where"]] = pd.DataFrame(
        proposals.tolist(), index=candidates.index
    )
    candidates["review_status"] = "PENDING_REVIEW"
    candidates["manual_subtype"] = ""
    candidates["manual_notes"] = ""
    candidates["teacher_subtype"] = ""
    candidates["teacher_confidence"] = ""
    candidates["final_subtype"] = ""
    candidates["image_dataset"] = "HuggingFaceM4/ChartQA"
    candidates["image_split"] = "val"
    candidates["image_index"] = candidates["dataset_index"]

    output_columns = [
        "dataset_index",
        "image_dataset",
        "image_split",
        "image_index",
        "transition_tag",
        "candidate_source",
        "question_base",
        "ground_truth",
        "prediction",
        "pred",
        "final_verdict",
        "judge_error_type",
        "judge_reason",
        "coarse_question_type",
        "proposed_subtype",
        "proposed_where",
        "review_status",
        "manual_subtype",
        "manual_notes",
        "teacher_subtype",
        "teacher_confidence",
        "final_subtype",
    ]
    candidates = candidates[output_columns].rename(
        columns={
            "question_base": "question",
            "prediction": "base_prediction",
            "pred": "sft_prediction",
        }
    )

    summary: dict[str, Any] = {
        "selection": {
            "all_validation_samples": int(len(frame)),
            "priority_both_wrong": int(frame["transition_tag"].eq("BOTH_WRONG").sum()),
            "priority_sft_regressions": int(
                frame["transition_tag"].eq("BASE_CORRECT_SFT_WRONG").sum()
            ),
            "review_candidates": int(len(candidates)),
        },
        "candidate_counts_by_transition": {
            key: int(value)
            for key, value in candidates["transition_tag"].value_counts().sort_index().items()
        },
        "candidate_counts_by_source": {
            key: int(value)
            for key, value in candidates["candidate_source"].value_counts().sort_index().items()
        },
        "proposed_subtype_counts": {
            key: int(value)
            for key, value in candidates["proposed_subtype"].value_counts().sort_index().items()
        },
        "taxonomy": SUBTYPES,
        "warning": (
            "proposed_subtype is heuristic triage only. final_subtype requires manual or teacher review."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "phase2c_visual_candidates.jsonl"
    json_path = args.output_dir / "phase2c_visual_summary.json"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in candidates.to_dict("records"):
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved review queue: {jsonl_path}")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
