"""Run or resume deterministic Qwen3-VL baseline inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finchart.config import load_config
from finchart.evaluator import deterministic_is_correct
from finchart.inference import load_baseline, run_dataset_inference
from finchart.utils import ensure_directories, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase1.yaml")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--force", action="store_true", help="Rerun Qwen even if a baseline CSV exists.")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_directories(args.results_dir)
    output = args.results_dir / f"{config.experiment_tag}_deterministic.csv"
    if output.exists() and not args.force:
        frame = pd.read_csv(output)
        print(f"Reusing saved baseline: {output} ({len(frame)} rows)")
        return
    set_seed(config.seed)
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    if config.shuffle_dataset:
        dataset = dataset.shuffle(seed=config.seed)
    dataset = dataset.select(range(config.dataset_offset, min(config.dataset_offset + config.num_samples, len(dataset))))
    if not len(dataset):
        raise ValueError("Empty dataset selection; check offset and sample count.")
    model, tokenizer = load_baseline(config)
    frame = run_dataset_inference(model, tokenizer, dataset, config)
    frame["deterministic_correct"] = frame.apply(lambda row: deterministic_is_correct(row.prediction, row.ground_truth, config.numeric_tolerance), axis=1)
    frame.to_csv(output, index=False)
    print(f"Saved baseline: {output}")
    print(f"Deterministic accuracy: {frame.deterministic_correct.mean():.2%}")


if __name__ == "__main__":
    main()
