#!/usr/bin/env python3
"""Export SFT visual-grounding failures and their ChartQA validation images.

This is an evaluation/diagnosis artifact only.  The exported cases come from
ChartQA val[0:500] and must never be used for SFT, DPO, or other training.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path

# Keep Hugging Face cache writes inside the writable, Git-ignored project tree.
PROJECT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_DIR / ".cache" / "datasets"))

from datasets import load_dataset


DATASET_NAME = "HuggingFaceM4/ChartQA"
DATASET_SPLIT = "val"
EXPECTED_STRICT_CASES = 31


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison",
        type=Path,
        default=PROJECT_DIR
        / "results"
        / "comparison"
        / "phase1_vs_sft_500_comparison.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR
        / "results"
        / "phase2c_visual_grounding_sft_failures",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def unwrap_answer(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def select_strict_visual_failures(rows: list[dict]) -> list[dict]:
    selected = []
    for row in rows:
        if bool(row.get("sft_deterministic_correct")):
            continue

        confirmed_remaining = (
            row.get("phase1_error_type") == "VISUAL_EXTRACTION"
            and row.get("confirmed_error_outcome") == "SFT_STILL_WRONG"
        )
        visual_regression = (
            row.get("question_type") == "VISUAL_EXTRACTION"
            and row.get("transition_tag") == "BASE_CORRECT_SFT_WRONG"
        )
        if not (confirmed_remaining or visual_regression):
            continue

        exported = dict(row)
        exported["visual_failure_tier"] = (
            "CONFIRMED_VISUAL_EXTRACTION_REMAINING"
            if confirmed_remaining
            else "HEURISTIC_VISUAL_REGRESSION"
        )
        selected.append(exported)

    selected.sort(key=lambda row: int(row["dataset_index"]))
    return selected


def encode_png(image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_gallery(rows: list[dict]) -> str:
    lines = [
        "# SFT visual-grounding failures",
        "",
        (
            "These cases come from frozen ChartQA `val[0:500]` and are for "
            "diagnosis only. **Do not use them for training or preference optimization.**"
        ),
        "",
        f"Cases: **{len(rows)}**",
        "",
    ]
    for row in rows:
        index = int(row["dataset_index"])
        lines.extend(
            [
                f"## val[{index}] — {row['visual_failure_tier']}",
                "",
                f"![ChartQA val {index}]({row['image_file']})",
                "",
                f"- Question: {row['question']}",
                f"- Ground truth: `{row['ground_truth']}`",
                f"- Base prediction: `{row['base_prediction']}`",
                f"- SFT prediction: `{row['sft_prediction']}`",
                f"- Transition: `{row['transition_tag']}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    comparison_path = args.comparison.resolve()
    output_dir = args.output_dir.resolve()
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_jsonl(comparison_path)
    selected = select_strict_visual_failures(source_rows)
    if len(source_rows) != 500:
        raise RuntimeError(f"Expected 500 comparison rows, found {len(source_rows)}")
    if len(selected) != EXPECTED_STRICT_CASES:
        raise RuntimeError(
            f"Expected {EXPECTED_STRICT_CASES} strict visual cases, found {len(selected)}"
        )

    chartqa_val = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    exported_rows = []
    for row in selected:
        index = int(row["dataset_index"])
        example = chartqa_val[index]
        source_question = str(
            example.get("query", example.get("question", ""))
        ).strip()
        source_answer = str(
            unwrap_answer(example.get("label", example.get("answer", "")))
        ).strip()
        if source_question != str(row["question"]).strip():
            raise RuntimeError(f"Question mismatch at ChartQA val[{index}]")
        if source_answer != str(row["ground_truth"]).strip():
            raise RuntimeError(f"Ground-truth mismatch at ChartQA val[{index}]")

        png_bytes = encode_png(example["image"])
        image_name = f"val_{index:04d}.png"
        image_path = images_dir / image_name
        image_path.write_bytes(png_bytes)
        width, height = example["image"].size

        exported = dict(row)
        exported.update(
            {
                "dataset": DATASET_NAME,
                "image_split": DATASET_SPLIT,
                "image_index": index,
                "image_file": f"images/{image_name}",
                "image_width": width,
                "image_height": height,
                "image_sha256": hashlib.sha256(png_bytes).hexdigest(),
                "evaluation_only": True,
                "allowed_for_training": False,
            }
        )
        exported_rows.append(exported)

    jsonl_path = output_dir / "sft_visual_grounding_failures_31.jsonl"
    json_path = output_dir / "sft_visual_grounding_failures_31.json"
    summary_path = output_dir / "sft_visual_grounding_failures_31_summary.json"
    gallery_path = output_dir / "gallery.md"
    write_jsonl(jsonl_path, exported_rows)
    json_path.write_text(
        json.dumps(exported_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    image_hashes = Counter(row["image_sha256"] for row in exported_rows)
    summary = {
        "source_comparison": str(comparison_path),
        "source_comparison_sha256": hashlib.sha256(
            comparison_path.read_bytes()
        ).hexdigest(),
        "dataset": DATASET_NAME,
        "split": DATASET_SPLIT,
        "frozen_range": "0:500",
        "selection": {
            "confirmed_visual_extraction_remaining": sum(
                row["visual_failure_tier"]
                == "CONFIRMED_VISUAL_EXTRACTION_REMAINING"
                for row in exported_rows
            ),
            "heuristic_visual_regressions": sum(
                row["visual_failure_tier"] == "HEURISTIC_VISUAL_REGRESSION"
                for row in exported_rows
            ),
            "total": len(exported_rows),
        },
        "case_images": len(exported_rows),
        "unique_chart_images_by_sha256": len(image_hashes),
        "evaluation_only": True,
        "allowed_for_training": False,
        "guardrail": "ChartQA validation cases must not enter SFT or DPO training.",
        "outputs": {
            "jsonl": jsonl_path.name,
            "json": json_path.name,
            "gallery": gallery_path.name,
            "images": "images/",
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    gallery_path.write_text(build_gallery(exported_rows), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
