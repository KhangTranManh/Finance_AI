#!/usr/bin/env python3
"""Evaluate the merged Phase 2C DPO model on frozen ChartQA val[0:500]."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


DATASET_NAME = "HuggingFaceM4/ChartQA"
SOURCE_SPLIT = "val"
FROZEN_EVAL_N = 500
FROZEN_MAX_NEW_TOKENS = 64
FROZEN_MAX_MODEL_LEN = 2048
PROMPT_TEMPLATE = (
    "Look carefully at the chart and answer the question.\n\n"
    "Question: {question}\n\n"
    "Return only the final answer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a contiguous partial JSONL instead of starting over.",
    )
    return parser.parse_args()


def unwrap_answer(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def normalize_phase1_answer(text) -> str:
    text = "" if text is None else str(text)
    text = text.lower().strip()
    text = re.sub(r"^\s*final\s+answer\s*:\s*", "", text)
    text = re.sub(r"^\s*answer\s*:\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*/\s*", "/", text)
    return text.strip(string.whitespace + ".,;:!?")


def try_parse_phase1_number(text):
    try:
        return float(
            normalize_phase1_answer(text).replace(",", "").replace("%", "").strip()
        )
    except (ValueError, TypeError):
        return None


def deterministic_phase1_match(prediction, ground_truth, tolerance=1e-6) -> bool:
    pred = normalize_phase1_answer(prediction)
    gt = normalize_phase1_answer(ground_truth)
    if pred == gt:
        return True
    pred_num = try_parse_phase1_number(pred)
    gt_num = try_parse_phase1_number(gt)
    return (
        pred_num is not None
        and gt_num is not None
        and abs(pred_num - gt_num) <= tolerance
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    model_dir = args.model.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing merged model config: {model_dir}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    jsonl_path = output_dir / "phase2c_dpo_val_0_500_predictions.jsonl"
    json_path = output_dir / "phase2c_dpo_val_0_500_predictions.json"
    summary_path = output_dir / "phase2c_dpo_val_0_500_summary.json"

    results: list[dict] = []
    if args.resume and jsonl_path.is_file():
        results = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        indices = [int(row["dataset_index"]) for row in results]
        if indices != list(range(len(results))) or len(results) > FROZEN_EVAL_N:
            raise RuntimeError("Refusing a non-contiguous or oversized resume JSONL")
        print(f"Resuming after {len(results)}/{FROZEN_EVAL_N} rows", flush=True)
        if len(results) == FROZEN_EVAL_N and summary_path.is_file():
            print(summary_path.read_text(encoding="utf-8"), flush=True)
            return

    dataset = load_dataset(DATASET_NAME, split=SOURCE_SPLIT).select(
        range(FROZEN_EVAL_N)
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    llm = LLM(
        model=str(model_dir),
        tokenizer=str(model_dir),
        dtype="bfloat16",
        max_model_len=FROZEN_MAX_MODEL_LEN,
        max_num_seqs=args.batch_size,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        seed=0,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=FROZEN_MAX_NEW_TOKENS,
        seed=0,
    )

    started = time.time()
    for start in range(len(results), FROZEN_EVAL_N, args.batch_size):
        stop = min(start + args.batch_size, FROZEN_EVAL_N)
        inputs = []
        metadata = []
        for dataset_index in range(start, stop):
            example = dataset[dataset_index]
            question = str(
                example.get("query", example.get("question", ""))
            ).strip()
            ground_truth = str(
                unwrap_answer(example.get("label", example.get("answer", "")))
            ).strip()
            image = example["image"].convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": PROMPT_TEMPLATE.format(question=question),
                        },
                    ],
                }
            ]
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image},
                }
            )
            metadata.append((dataset_index, question, ground_truth))

        outputs = llm.generate(inputs, sampling_params=sampling, use_tqdm=False)
        if len(outputs) != len(metadata):
            raise RuntimeError(
                f"Expected {len(metadata)} outputs, received {len(outputs)}"
            )
        for (dataset_index, question, ground_truth), output in zip(metadata, outputs):
            prediction = output.outputs[0].text.strip()
            results.append(
                {
                    "dataset_index": dataset_index,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "deterministic_correct": deterministic_phase1_match(
                        prediction, ground_truth
                    ),
                }
            )
        write_jsonl(jsonl_path, results)
        correct = sum(row["deterministic_correct"] for row in results)
        print(
            f"Evaluated {stop}/{FROZEN_EVAL_N}; "
            f"running accuracy={correct / len(results):.2%}",
            flush=True,
        )

    elapsed = time.time() - started
    correct = sum(row["deterministic_correct"] for row in results)
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = {
        "dataset": DATASET_NAME,
        "split": SOURCE_SPLIT,
        "range": "0:500",
        "model": str(model_dir),
        "engine": "vLLM",
        "merged_adapter": True,
        "merge_dtype": "bfloat16",
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest(),
        "decoding": {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": FROZEN_MAX_NEW_TOKENS,
            "max_model_len": FROZEN_MAX_MODEL_LEN,
            "seed": 0,
        },
        "matcher": "phase1_exact_or_numeric_tolerance_1e-6",
        "total": len(results),
        "correct": correct,
        "incorrect": len(results) - correct,
        "accuracy": correct / len(results),
        "elapsed_seconds": round(elapsed, 3),
        "outputs": {
            "jsonl": str(jsonl_path),
            "json": str(json_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
