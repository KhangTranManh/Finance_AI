"""Judge deterministic mismatches, checkpointing after every configured case."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finchart.config import load_config
from finchart.judge import JUDGE_COLUMNS, call_judge_with_retry
from finchart.metrics import build_final_verdict
from finchart.utils import ensure_directories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase1.yaml")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if not config.judge_enabled:
        raise RuntimeError("Set judge credentials in your environment; see .env.example.")
    baseline_path = args.results_dir / f"{config.experiment_tag}_deterministic.csv"
    checkpoint_path = args.checkpoint_dir / f"{config.experiment_tag}_judge_checkpoint.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Run scripts/run_baseline.py first: {baseline_path}")
    ensure_directories(args.checkpoint_dir)
    frame = pd.read_csv(baseline_path)
    if checkpoint_path.exists():
        old = pd.read_csv(checkpoint_path)
        if len(old) != len(frame):
            raise ValueError("Judge checkpoint length does not match baseline.")
        for column, default in JUDGE_COLUMNS.items():
            frame[column] = old[column] if column in old else default
    else:
        for column, default in JUDGE_COLUMNS.items():
            frame[column] = default
    frame.loc[frame.deterministic_correct, "judge_status"] = "NOT_NEEDED"
    pending = ~frame.deterministic_correct & (frame.judge_status != "SUCCESS")
    frame.loc[pending, "judge_status"] = "PENDING"
    dataset = load_dataset(config.dataset_name, split=config.dataset_split).select(range(config.dataset_offset, config.dataset_offset + len(frame)))
    indices = frame.index[frame.judge_status == "PENDING"].tolist()
    if args.max_cases is not None:
        indices = indices[:args.max_cases]
    for index in indices:
        sample = dataset[int(frame.at[index, "id"])]
        result = call_judge_with_retry(sample["image"], frame.at[index, "question"], frame.at[index, "ground_truth"], frame.at[index, "prediction"], config)
        for key, value in result.items():
            frame.at[index, key] = value
        if config.save_after_each_judge:
            frame.to_csv(checkpoint_path, index=False)
        time.sleep(config.judge_pause_seconds)
    frame["final_verdict"] = frame.apply(build_final_verdict, axis=1)
    frame.to_csv(checkpoint_path, index=False)
    print(f"Judge checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    main()
