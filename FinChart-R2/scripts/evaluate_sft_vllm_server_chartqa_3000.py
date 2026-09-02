#!/usr/bin/env python3
"""Evaluate FinChart SFT through a local vLLM OpenAI-compatible server.

The run contains two separately reported evaluation slices:
  * frozen ChartQA val[0:500]
  * full ChartQA test[0:2500]

No training examples are mixed into this evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import re
import string
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
from datasets import load_dataset


DATASET_NAME = "HuggingFaceM4/ChartQA"
MODEL_NAME = "FinChart-SFT-408"
SOURCE_ADAPTER = "Kxck/Finance_500_v1"
OUTPUT_STEM = "sft_vllm_chartqa_val500_test2500"
MAX_NEW_TOKENS = 64
PROMPT_TEMPLATE = (
    "Look carefully at the chart and answer the question.\n\n"
    "Question: {question}\n\n"
    "Return only the final answer."
)
SLICES = (("val", 500), ("test", 2500))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8999/v1")
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def unwrap_answer(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def normalize_answer(text) -> str:
    text = "" if text is None else str(text)
    text = text.lower().strip()
    text = re.sub(r"^\s*final\s+answer\s*:\s*", "", text)
    text = re.sub(r"^\s*answer\s*:\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*/\s*", "/", text)
    return text.strip(string.whitespace + ".,;:!?")


def try_number(text):
    try:
        return float(normalize_answer(text).replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return None


def deterministic_match(prediction, ground_truth, tolerance=1e-6) -> bool:
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    if prediction == ground_truth:
        return True
    pred_num, gt_num = try_number(prediction), try_number(ground_truth)
    return (
        pred_num is not None
        and gt_num is not None
        and abs(pred_num - gt_num) <= tolerance
    )


def image_data_url(image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=False)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


async def request_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    split: str,
    index: int,
    example: dict,
    max_retries: int,
) -> dict:
    question = str(example.get("query", example.get("question", ""))).strip()
    ground_truth = str(
        unwrap_answer(example.get("label", example.get("answer", "")))
    ).strip()
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(example["image"])},
                    },
                    {
                        "type": "text",
                        "text": PROMPT_TEMPLATE.format(question=question),
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": MAX_NEW_TOKENS,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            started = time.perf_counter()
            async with semaphore:
                response = await client.post(
                    "/chat/completions", json=payload, headers=headers
                )
            response.raise_for_status()
            body = response.json()
            prediction = str(body["choices"][0]["message"]["content"]).strip()
            usage = body.get("usage") or {}
            return {
                "dataset": DATASET_NAME,
                "split": split,
                "dataset_index": index,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "deterministic_correct": deterministic_match(
                    prediction, ground_truth
                ),
                "finish_reason": body["choices"][0].get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "latency_seconds": round(time.perf_counter() - started, 4),
            }
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            last_error = error
            if attempt >= max_retries:
                break
            await asyncio.sleep(min(2**attempt, 15))
    raise RuntimeError(f"Failed {split}[{index}] after retries: {last_error}")


async def main_async() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{OUTPUT_STEM}_predictions.jsonl"
    json_path = output_dir / f"{OUTPUT_STEM}_predictions.json"
    summary_path = output_dir / f"{OUTPUT_STEM}_summary.json"
    api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError("API key file is empty")

    rows = []
    if not args.no_resume and jsonl_path.is_file():
        rows = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    completed = {(row["split"], int(row["dataset_index"])) for row in rows}
    if len(completed) != len(rows):
        raise RuntimeError("Resume JSONL contains duplicate split/index keys")

    datasets_by_split = {
        split: load_dataset(DATASET_NAME, split=split) for split, _ in SLICES
    }
    jobs = []
    for split, count in SLICES:
        dataset = datasets_by_split[split]
        if len(dataset) < count:
            raise RuntimeError(f"{split} contains {len(dataset)} rows, expected {count}")
        for index in range(count):
            if (split, index) not in completed:
                jobs.append((split, index, dataset[index]))

    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )
    timeout = httpx.Timeout(args.request_timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    started_at = time.time()
    async with httpx.AsyncClient(
        base_url=args.base_url, limits=limits, timeout=timeout
    ) as client:
        tasks = [
            asyncio.create_task(
                request_one(
                    client,
                    semaphore,
                    api_key,
                    split,
                    index,
                    example,
                    args.max_retries,
                )
            )
            for split, index, example in jobs
        ]
        for completed_count, task in enumerate(asyncio.as_completed(tasks), 1):
            rows.append(await task)
            if completed_count % 25 == 0 or completed_count == len(tasks):
                rows.sort(key=lambda row: (row["split"], row["dataset_index"]))
                write_jsonl_atomic(jsonl_path, rows)
                correct = sum(bool(row["deterministic_correct"]) for row in rows)
                print(
                    f"Completed new {completed_count}/{len(tasks)}; "
                    f"stored {len(rows)}/3000; running accuracy={correct/len(rows):.2%}",
                    flush=True,
                )

    rows.sort(key=lambda row: (row["split"], row["dataset_index"]))
    json_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    per_split = {}
    for split, count in SLICES:
        split_rows = [row for row in rows if row["split"] == split]
        correct = sum(bool(row["deterministic_correct"]) for row in split_rows)
        per_split[split] = {
            "expected": count,
            "total": len(split_rows),
            "correct": correct,
            "incorrect": len(split_rows) - correct,
            "accuracy": correct / len(split_rows),
        }
    total_correct = sum(bool(row["deterministic_correct"]) for row in rows)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_NAME,
        "model": MODEL_NAME,
        "source_adapter": SOURCE_ADAPTER,
        "engine": "vLLM OpenAI-compatible server",
        "evaluation_slices": {"val": "0:500", "test": "0:2500"},
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest(),
        "decoding": {
            "temperature": 0.0,
            "max_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
        },
        "matcher": "phase1_exact_or_numeric_tolerance_1e-6",
        "per_split": per_split,
        "combined_descriptive_only": {
            "total": len(rows),
            "correct": total_correct,
            "incorrect": len(rows) - total_correct,
            "accuracy": total_correct / len(rows),
        },
        "elapsed_seconds_this_invocation": round(time.time() - started_at, 3),
        "finish_reasons": dict(Counter(str(row["finish_reason"]) for row in rows)),
        "guardrails": [
            "Validation and test predictions are evaluation-only.",
            "Per-split metrics are primary; combined accuracy is descriptive only.",
            "No ChartQA train examples are included.",
        ],
        "outputs": {"jsonl": jsonl_path.name, "json": json_path.name},
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main_async())
