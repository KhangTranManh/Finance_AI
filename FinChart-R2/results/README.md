# Local generated results

This directory is ignored by Git except for this file. It contains only local datasets and intermediate evidence needed to reproduce or audit the current experiments.

## Retained artifacts

### Phase 2A

```text
phase2a_pilot_500_v3_train_clean.jsonl
phase2a_pilot_500_v3_train_clean_report.json
phase2a_pilot_500_v3_manual_audit_approval.json
```

The JSONL contains 408 strict-clean examples used by the Phase 2B pilot.

### Phase 2B comparison

```text
comparison/phase1_vs_sft_500_comparison.jsonl
comparison/phase1_vs_sft_500_summary.json
```

These files preserve per-example base/SFT transition tags and the aggregate comparison. The approved, tracked summary is [the Phase 2B report](../../reports/phase2b_pilot_408_evaluation.md).

### Phase 2C preference construction

```text
finchart_r2_phase2c_train_mining/
  dpo_train_error.jsonl
  phase2c_train_500_2500_manifest.json
  dpo_train_teacher_raw_capture.jsonl
  dpo_train_teacher_raw_capture_report.json
  phase2c_teacher_v1_analysis_report.json
  phase2c_teacher_v1_dpo_candidates_provisional.jsonl
```

- `dpo_train_error.jsonl`: 906 deterministic SFT mismatch candidates from ChartQA `train[500:2500]`.
- `dpo_train_teacher_raw_capture.jsonl`: raw, resumable teacher responses for all 906 candidates.
- `phase2c_teacher_v1_dpo_candidates_provisional.jsonl`: 377 automatically gated prompt/chosen/rejected pairs.

The 377 pairs still require manual audit. They must not be presented as ground truth or used as a final training dataset without that audit.

## Intentionally removed

Duplicate CSV exports, deterministic-correct mining rows, deprecated strict-schema audit outputs, validation-derived preference diagnostics, failed API probes, and regenerable routing tables are not retained. They either duplicate the artifacts above, are obsolete, or cannot be used for reportable training.

Do not commit credentials, provider reasoning/logs, raw datasets, adapters, or checkpoints.
