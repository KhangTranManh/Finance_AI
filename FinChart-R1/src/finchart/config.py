"""Configuration loading for the Phase 1 experiment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class Phase1Config:
    base_model: str = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"
    load_in_4bit: bool = True
    max_new_tokens: int = 64
    do_sample: bool = False
    seed: int = 42
    dataset_name: str = "HuggingFaceM4/ChartQA"
    dataset_split: str = "val"
    num_samples: int = 500
    dataset_offset: int = 0
    shuffle_dataset: bool = False
    numeric_tolerance: float = 1e-6
    judge_timeout_seconds: int = 120
    judge_max_retries: int = 3
    judge_backoff_base_seconds: float = 1.0
    judge_backoff_max_seconds: float = 8.0
    judge_pause_seconds: float = 0.2
    judge_min_confidence: float = 0.70
    save_after_each_judge: bool = True
    judge_api_key: str = field(default_factory=lambda: os.getenv("JUDGE_API_KEY", ""))
    judge_base_url: str = field(default_factory=lambda: os.getenv("JUDGE_BASE_URL", ""))
    judge_model: str = field(default_factory=lambda: os.getenv("JUDGE_MODEL", ""))

    @property
    def judge_enabled(self) -> bool:
        return bool(self.judge_api_key and self.judge_base_url and self.judge_model)

    @property
    def experiment_tag(self) -> str:
        return f"qwen3vl4b_chartqa_{self.dataset_split}_{self.dataset_offset}_{self.num_samples}"


def load_config(path: str | Path) -> Phase1Config:
    """Load a YAML config and overlay credentials from environment variables."""
    config_path = Path(path).resolve()
    load_dotenv(config_path.parents[1] / ".env")
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    baseline = raw.get("baseline", {})
    dataset = raw.get("dataset", {})
    evaluator = raw.get("evaluator", {})
    judge = raw.get("judge", {})
    return Phase1Config(
        base_model=baseline.get("model", Phase1Config.base_model),
        load_in_4bit=baseline.get("load_in_4bit", True),
        max_new_tokens=baseline.get("max_new_tokens", 64),
        do_sample=baseline.get("do_sample", False),
        seed=baseline.get("seed", 42),
        dataset_name=dataset.get("name", Phase1Config.dataset_name),
        dataset_split=dataset.get("split", "val"),
        num_samples=dataset.get("num_samples", 500),
        dataset_offset=dataset.get("offset", 0),
        shuffle_dataset=dataset.get("shuffle", False),
        numeric_tolerance=evaluator.get("numeric_tolerance", 1e-6),
        judge_timeout_seconds=judge.get("timeout_seconds", 120),
        judge_max_retries=judge.get("max_retries", 3),
        judge_backoff_base_seconds=judge.get("backoff_base_seconds", 1.0),
        judge_backoff_max_seconds=judge.get("backoff_max_seconds", 8.0),
        judge_pause_seconds=judge.get("pause_seconds", 0.2),
        judge_min_confidence=judge.get("min_confidence", 0.70),
        save_after_each_judge=judge.get("save_after_each", True),
    )
