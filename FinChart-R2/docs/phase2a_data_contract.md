# Phase 2A data contract

Phase 2A produces teacher-assisted, concise structured supervision from the ChartQA **training** split only. The frozen Phase 1 validation subset is excluded from every annotation and training stage.

## Annotation contract

Every teacher response must parse as JSON and contain `task_type`, `subtype`, `target_series`, `target_category`, `relevant_values`, `operation`, `calculation`, `final_answer`, and `confidence`.

`task_type` is one of `numerical_reasoning`, `visual_grounding`, `counting`, or `logical_reasoning`. `operation` is one of `none`, `lookup`, `sum`, `difference`, `average`, `median`, `ratio`, `percentage`, `percentage_change`, `count`, `comparison`, `max_difference`, `min_max`, or `multi_step`.

Validation checks schema and taxonomy membership, confidence, answer agreement with ChartQA (including numeric and percentage/proportion equivalence), semantic conflicts, and deterministic arithmetic where possible. Statuses are `VALIDATED`, `REVIEW_REPRESENTATION`, `REVIEW_CONFLICT`, `LOW_CONFIDENCE`, `INVALID_SCHEMA`, and `TEACHER_ERROR`.

## Pilot dataset policy

The 500-example pilot is a pipeline and training-hypothesis check, not the final dataset. `approve_and_build_train_v3.py` retains only examples that are `VALIDATED`, have confidence at least 0.90, and have a task/operation-compatible annotation. The resulting local `v3_train_clean` dataset currently has 408 examples.

All raw outputs remain local under `results/`. To reproduce the pilot, configure `.env` from `.env.template`, then run the three Phase 2A scripts in the root README.
