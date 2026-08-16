"""Coverage-aware Phase 1 metrics."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_final_verdict(row: pd.Series) -> str:
    if bool(row["deterministic_correct"]):
        return "CORRECT"
    if row.get("judge_status") == "SUCCESS" and row.get("judge_verdict") in {
        "CORRECT", "INCORRECT", "AMBIGUOUS", "POSSIBLE_LABEL_ERROR"
    }:
        return str(row["judge_verdict"])
    return "JUDGE_ERROR" if row.get("judge_status") == "JUDGE_ERROR" else "UNJUDGED"


def calculate_summary(frame: pd.DataFrame, experiment_tag: str, config: Any) -> dict[str, Any]:
    """Return metrics while retaining unresolved and technical-failure counts."""
    total = len(frame)
    counts = frame["final_verdict"].value_counts().to_dict()
    correct, incorrect = int(counts.get("CORRECT", 0)), int(counts.get("INCORRECT", 0))
    resolved = correct + incorrect
    success = int((frame["judge_status"] == "SUCCESS").sum())
    failures = int((frame["judge_status"] == "JUDGE_ERROR").sum())
    return {
        "experiment_tag": experiment_tag,
        "base_model": config.base_model,
        "dataset": config.dataset_name,
        "dataset_split": config.dataset_split,
        "dataset_offset": config.dataset_offset,
        "samples": total,
        "deterministic_correct": int(frame["deterministic_correct"].sum()),
        "deterministic_accuracy": float(frame["deterministic_correct"].mean()) if total else None,
        "final_correct": correct,
        "final_incorrect": incorrect,
        "ambiguous": int(counts.get("AMBIGUOUS", 0)),
        "possible_label_error": int(counts.get("POSSIBLE_LABEL_ERROR", 0)),
        "judge_error": int(counts.get("JUDGE_ERROR", 0)),
        "unjudged": int(counts.get("UNJUDGED", 0)),
        "resolved_samples": resolved,
        "resolved_accuracy": correct / resolved if resolved else None,
        "resolved_coverage": resolved / total if total else None,
        "judge_success": success,
        "judge_technical_errors": failures,
        "judge_technical_success_rate": success / (success + failures) if success + failures else None,
    }


def error_distribution(frame: pd.DataFrame) -> pd.Series:
    successful = frame[frame["judge_status"] == "SUCCESS"]
    return successful.loc[successful["judge_error_type"].notna() & (successful["judge_error_type"] != "NONE"), "judge_error_type"].value_counts()
