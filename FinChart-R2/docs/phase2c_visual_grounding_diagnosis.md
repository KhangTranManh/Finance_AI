# Phase 2C — Visual Grounding Diagnosis

Phase 2C is a diagnostic stage, not an additional training run. It determines whether the visual and counting failures that remain after the 408-example SFT pilot are sufficiently localizable and frequent to justify grounding-specific supervision.

## Candidate selection

The analysis joins the Phase 1 base deterministic CSV, the Phase 1 semantic-audited final CSV, and the SFT validation CSV by the frozen ChartQA validation index.

It prioritizes these deterministic transitions:

- `BOTH_WRONG`: both base and SFT fail.
- `BASE_CORRECT_SFT_WRONG`: SFT regression.

Candidates are retained when they are a Phase 1 confirmed `VISUAL_EXTRACTION` or `COUNTING` error, or when a transparent question-text heuristic identifies a visual/counting question. The latter is only triage; it is not a semantic label.

## Review taxonomy

Manual review or a stronger offline teacher must select exactly one subtype:

- `WRONG_SERIES`
- `WRONG_COLOR`
- `WRONG_CATEGORY`
- `WRONG_VALUE`
- `WRONG_POINT`
- `LEGEND_ASSOCIATION`
- `AXIS_ALIGNMENT`
- `COUNTING_ERROR`
- `EXTREMA_ERROR`
- `CROP_SMALL_TEXT`
- `OTHER_VISUAL`

The generated candidate file includes `proposed_subtype`, `proposed_where`, `manual_subtype`, `teacher_subtype`, `final_subtype`, and `review_status`. Only the Phase 1 error type is authoritative at export time. Proposed subtypes are rules to prioritize review, not labels to train on.

## Teacher-assisted annotation

`annotate_phase2c_visual_failures.py` sends only selected failure cases to a vision teacher. The request includes the chart image, question, ChartQA ground-truth answer, base prediction, SFT prediction, and the existing Phase 1/2C triage metadata. The teacher returns a closed-taxonomy visual label plus an optional normalized bounding box:

```json
{
  "error_type": "VISUAL_GROUNDING",
  "subtype": "WRONG_SERIES",
  "target_series": "Blue",
  "target_category": "2018",
  "target_color": "Blue",
  "relevant_value": 52,
  "bbox": [100, 200, 400, 700],
  "confidence": 0.94
}
```

Bounding boxes are `[ymin, xmin, ymax, xmax]`, normalized to 0–1000. The script validates JSON/schema/taxonomy/confidence and puts a stratified 20% sample into manual audit. Teacher output is a candidate label only; it is never ground truth and is never used directly for training without audit.

## Outputs

Run:

```powershell
python FinChart-R2/scripts/build_phase2c_visual_diagnosis.py
```

The scripts write local-only JSON artifacts under `FinChart-R2/results/phase2c_visual_diagnosis/`:

- `phase2c_visual_candidates.jsonl`: per-chart review queue.
- `phase2c_visual_summary.json`: candidate counts by transition, source, and proposed subtype.
- `phase2c_teacher_annotations.jsonl`: resumable teacher candidate labels.
- `phase2c_teacher_audit.jsonl`: manual-audit queue.
- `phase2c_teacher_report.json`: validation and audit summary.

These files are ignored by Git because they contain raw predictions and review annotations.

## Decision gate

Grounding-specific supervision is worth adding only if manual/teacher review finds a repeated, actionable concentration of visual subtypes (for example, wrong value, legend association, wrong series, or axis alignment). Otherwise, Phase 2 should first improve concise SFT target formatting or data balance rather than introducing a detector or grounding model prematurely.
