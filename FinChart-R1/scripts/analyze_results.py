"""Build final Phase 1 CSV and coverage-aware summary from a judge checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finchart.config import load_config
from finchart.metrics import build_final_verdict, calculate_summary, error_distribution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase1.yaml")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint_dir / f"{config.experiment_tag}_judge_checkpoint.csv"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No judge checkpoint found: {checkpoint}")
    frame = pd.read_csv(checkpoint)
    frame["final_verdict"] = frame.apply(build_final_verdict, axis=1)
    summary = calculate_summary(frame, config.experiment_tag, config)
    final_csv = args.results_dir / f"{config.experiment_tag}_phase1_final.csv"
    summary_json = args.results_dir / f"{config.experiment_tag}_phase1_summary.json"
    frame.to_csv(final_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Error distribution:\n", error_distribution(frame).to_string())


if __name__ == "__main__":
    main()
