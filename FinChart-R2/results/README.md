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
  phase2c_teacher_v1_analysis_all.jsonl
  phase2c_teacher_v1_analysis_report.json
  phase2c_teacher_v1_resolved_correct.jsonl
  phase2c_teacher_v1_dpo_candidates_provisional.jsonl
  phase2c_teacher_v1_dpo_candidates_diversity_report.json
  phase2c_teacher_v1_review.jsonl
```

- `dpo_train_error.jsonl`: 906 deterministic SFT mismatch candidates from ChartQA `train[500:2500]`.
- `dpo_train_teacher_raw_capture.jsonl`: raw, resumable teacher responses for all 906 candidates.
- `phase2c_teacher_v1_dpo_candidates_provisional.jsonl`: 386 automatically gated prompt/chosen/rejected pairs.
- `phase2c_teacher_v1_dpo_candidates_diversity_report.json`: reproducible heuristic task, operation, and answer-type distributions.
- `phase2c_teacher_v1_resolved_correct.jsonl`: 326 representation-equivalent or already-correct cases excluded from DPO.
- `phase2c_teacher_v1_review.jsonl`: 194 ambiguity, teacher/reference-conflict, or unknown-verdict cases requiring review.

The 386 pairs still require manual audit. They must not be presented as ground truth or used as a final training dataset without that audit.

### Phase 2C diagnostic DPO training and evaluation

```text
checkpoints/dpo_386_provisional/
  adapter_config.json
  MODEL_CARD.md
  trainer_state.json
  training_manifest.json

vali/phase2c_dpo_vllm_val_0_500/
  phase2c_dpo_val_0_500_predictions.jsonl
  phase2c_dpo_val_0_500_predictions.json
  phase2c_dpo_val_0_500_summary.json
  reproduction_manifest.json
```

The published adapter is `Kxck/Finance_500_v1_DPO_386_provisional`. The local metadata records the 347/39 train/evaluation split and completed 44-step one-epoch run. The historical standalone vLLM evaluation contains exactly 500 ordered JSON records and scores 324/500 (64.8%) with the unchanged Phase 1 exact/numeric matcher. It is retained for reproduction, but the later HTTP-server comparison below is the primary matched SFT/DPO result. See the [historical Phase 2C report](../../reports/phase2c_dpo_386_vllm_evaluation.md).

### SFT vLLM validation and test evaluation

```text
vali/sft_vllm_chartqa_3000/
  sft_vllm_chartqa_val500_test2500_predictions.jsonl
  sft_vllm_chartqa_val500_test2500_predictions.json
  sft_vllm_chartqa_val500_test2500_summary.json
```

The merged-BF16 SFT-only run contains exactly 3,000 evaluation records: ChartQA `val[0:500]` scores 327/500 (65.4%), while `test[0:2500]` scores 1,824/2,500 (72.96%). The combined 2,151/3,000 (71.70%) value is descriptive only; the official splits are reported separately. No ChartQA train examples are included. The tracked interpretation and reproducibility contract are in the [SFT vLLM report](../../reports/phase2b_sft_vllm_chartqa_3000_evaluation.md).

### Matched SFT+DPO vLLM evaluation and paired comparison

```text
vali/dpo_vllm_chartqa_3000/
  dpo_vllm_chartqa_val500_test2500_predictions.jsonl
  dpo_vllm_chartqa_val500_test2500_predictions.json
  dpo_vllm_chartqa_val500_test2500_summary.json

comparison/sft_vs_dpo_val500_test2500/
  sft_vs_dpo_val_paired.jsonl
  sft_vs_dpo_test_paired.jsonl
  sft_vs_dpo_val500_test2500_report.json
```

Under the same merged-BF16 vLLM HTTP-serving path, SFT+DPO scores 322/500 (64.4%) on validation and 1,822/2,500 (72.88%) on test. SFT scores 327/500 and 1,824/2,500 respectively, so DPO is lower by five and two answers. Test transitions are 1,812 both-correct, 666 both-wrong, 12 SFT-only correct, and 10 DPO-only correct. See the [matched comparison report](../../reports/phase2c_sft_dpo_val500_test2500_comparison.md).

### Phase 2D test-2500 failure analysis

```text
phase2d_sft_failures_test_2500/
  sft_test_2500_predictions_analyzed.jsonl
  sft_test_2500_failures_676.jsonl
  sft_test_2500_failures_676.json
  sft_test_2500_teacher_priority_after_local_gates.jsonl
  sft_test_2500_failure_report.json
```

This pre-teacher diagnostic extracts all 676 strict SFT failures from ChartQA `test[0:2500]`. Deterministic local gates identify 43 format-recoverable cases and 3 percentage/proportion candidates, leaving 630 teacher-priority mismatches. The heuristic failure mix is 288 numerical, 229 visual-grounding/lookup, 126 counting, and 33 logical. These labels are not ground truth. Every record is `evaluation_only=true` and `allowed_for_training=false`; see the [Phase 2D analysis report](../../reports/phase2d_sft_test2500_failure_analysis.md).

### Val-500 versus test-2500 distribution comparison

```text
phase2d_sft_val500_test2500_distribution/
  sft_val_distribution_labeled.jsonl
  sft_test_distribution_labeled.jsonl
  sft_val500_test2500_distribution_report.json
```

Every validation and test row is tagged with heuristic `task_type`, `answer_type`, and `operation_type` fields. Visual grounding dominates volume in both splits, while numerical reasoning dominates failures. Test has substantially more numeric answers, lookup operations, and counting questions than validation, while logical/yes-no coverage is smaller. See the [distribution report](../../reports/phase2d_sft_val500_test2500_distribution.md).

### SFT visual-grounding failure extraction

```text
phase2c_visual_grounding_sft_failures/
  images/                                      31 question-level PNG files
  sft_visual_grounding_failures_31.jsonl
  sft_visual_grounding_failures_31.json
  sft_visual_grounding_failures_31_summary.json
  gallery.md
```

This diagnostic subset contains 26 confirmed visual-extraction failures that SFT did not fix and 5 heuristic visual regressions where Base was correct but SFT was wrong. The 31 question-level cases map to 30 unique chart images. Every record is marked `evaluation_only=true` and `allowed_for_training=false` because it comes from frozen ChartQA `val[0:500]`.

## Intentionally removed

Duplicate CSV exports, deterministic-correct mining rows, deprecated strict-schema audit outputs, validation-derived preference diagnostics, failed API probes, and superseded routing tables are not retained. They either duplicate the artifacts above, are obsolete, or cannot be used for reportable training.

Do not commit credentials, provider reasoning/logs, raw datasets, adapters, or checkpoints.
