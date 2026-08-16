"""Qwen3-VL baseline loading and inference."""

from __future__ import annotations

from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from .config import Phase1Config
from .evaluator import deterministic_is_correct


def load_baseline(config: Phase1Config) -> tuple[Any, Any]:
    """Load the notebook's Qwen3-VL checkpoint through Unsloth."""
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        config.base_model,
        load_in_4bit=config.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer


def build_baseline_messages(image: Any, question: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": (
            "Look carefully at the chart and answer the question.\n\n"
            f"Question: {question}\n\nReturn only the final answer."
        )},
    ]}]


def run_baseline_inference(model: Any, tokenizer: Any, image: Any, question: str, config: Phase1Config) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(
        build_baseline_messages(image, question), add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(image, prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            use_cache=True,
        )
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def get_ground_truth(sample: dict[str, Any]) -> str:
    label = sample["label"]
    return str(label[0]) if isinstance(label, list) else str(label)


def run_dataset_inference(model: Any, tokenizer: Any, dataset: Any, config: Phase1Config) -> pd.DataFrame:
    """Run deterministic greedy baseline inference over a selected dataset subset."""
    rows: list[dict[str, Any]] = []
    for local_id, sample in enumerate(tqdm(dataset, desc="Baseline inference")):
        prediction = run_baseline_inference(model, tokenizer, sample["image"], sample["query"], config)
        ground_truth = get_ground_truth(sample)
        rows.append({
            "id": local_id,
            "dataset_index": config.dataset_offset + local_id,
            "question": sample["query"],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "deterministic_correct": deterministic_is_correct(
                prediction, ground_truth, config.numeric_tolerance
            ),
            "human_or_machine": sample.get("human_or_machine"),
        })
    return pd.DataFrame(rows)
