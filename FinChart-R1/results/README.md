# Local results

Place your experiment artifacts in this directory. CSV and JSON result files are ignored by Git so raw predictions, judge responses, and local API metadata are not published accidentally.

The Phase 1 scripts conventionally create:

- `qwen3vl4b_chartqa_val_<offset>_<samples>_deterministic.csv`
- `checkpoints/qwen3vl4b_chartqa_val_<offset>_<samples>_judge_checkpoint.csv`
- `qwen3vl4b_chartqa_val_<offset>_<samples>_phase1_final.csv`
- `qwen3vl4b_chartqa_val_<offset>_<samples>_phase1_summary.json`

If you decide to publish a representative sample later, review it for API metadata and sensitive data first, then explicitly force-add only that curated file.
