# Local generated outputs

This directory is deliberately excluded from Git except for this note. It may contain teacher checkpoints, request logs, audits, candidate datasets, approved pilot SFT data, and evaluations.

Recreate the Phase 2A pilot locally with the scripts listed in the root README. Do not commit raw teacher responses, provider metadata, API logs, or generated training data.

At the current pilot stage, `phase2a_pilot_500_v3_train_clean.jsonl` is the only retained training artifact. It contains the strict-clean examples approved for a small SFT experiment; it is not a final full-train dataset.

`scripts/compare_phase1_vs_sft.py` writes local JSONL/JSON comparison artifacts under `comparison/`. They contain per-sample transition tags between base and SFT plus a summary of confirmed Phase 1 errors fixed by SFT. These artifacts are also ignored; the approved aggregate result is tracked in `../../reports/phase2b_pilot_408_evaluation.md`.

`scripts/build_phase2c_visual_diagnosis.py` writes a local visual/counting review queue under `phase2c_visual_diagnosis/`. It intentionally leaves final subtypes pending manual or teacher review.

## Phase 2C train-only preference-mining outputs

[Notebook 04](../notebooks/04_FinChart_R2_Phase2C_DPO_Train_Preference_Mining_Colab.ipynb) writes resumable generated artifacts to the configured Google Drive run directory:

~~~text
phase2c/train_preference_mining/
  phase2c_train_500_2500_sft_predictions.jsonl
  phase2c_train_500_2500_sft_errors.jsonl
  phase2c_train_500_2500_sft_correct.jsonl
  phase2c_train_500_2500_manifest.json
~~~

The error JSONL is a train-only teacher-review queue, not direct DPO supervision. Keep the raw predictions and audit metadata local. Only a later validated pair artifact with response-schema-matched prompt, chosen, and rejected fields may be used by Phase 2C DPO.

The earlier 51-pair validation-derived DPO output is an infrastructure diagnostic only. Do not commit it as a benchmark artifact or evaluate its adapter on the same frozen validation subset.
