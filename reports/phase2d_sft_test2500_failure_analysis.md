# Phase 2D pre-teacher analysis: SFT on ChartQA test[0:2500]

## Outcome

The SFT-only adapter `Kxck/Finance_500_v1`, served as a merged BF16 checkpoint through vLLM, answers 1,824/2,500 ChartQA test questions correctly (72.96%) and fails 676 (27.04%). This analysis is the deterministic filtering stage before any vision-language teacher call.

| Heuristic task family | Total questions | Strict failures | Error rate | Remaining after local gates |
| --- | ---: | ---: | ---: | ---: |
| Numerical reasoning | 698 | 288 | 41.26% | 257 |
| Visual grounding / lookup | 1,196 | 229 | 19.15% | 220 |
| Counting | 502 | 126 | 25.10% | 120 |
| Logical reasoning | 104 | 33 | 31.73% | 33 |
| **Total** | **2,500** | **676** | **27.04%** | **630** |

Numerical reasoning is the largest failure family by both count and heuristic error rate. Visual grounding is the second-largest block by count. These task labels come from question/answer patterns and are routing heuristics, not teacher-verified error causes. A numerical or counting failure may still originate from incorrect visual extraction.

## Local gates before teacher use

Of the 676 strict mismatches:

- 43 contain a locally extractable answer that matches the reference, so they are format-recoverable candidates.
- 3 are percentage/proportion scale-equivalence candidates.
- 630 remain as teacher-priority mismatches.
- 104 emit structured SFT fields despite the final-answer-only prompt.
- 55 reach the 64-token completion limit.

The last two signals overlap with other categories. They are diagnostic flags rather than mutually exclusive routes.

## Dominant operations among failures

| Heuristic operation | Failures |
| --- | ---: |
| Lookup | 175 |
| Count | 126 |
| Difference | 82 |
| Average | 72 |
| Sum | 45 |
| Extrema | 44 |
| Ratio | 38 |
| Percentage | 36 |
| Comparison | 33 |

Human-authored questions are a clear weakness: 482/1,250 fail (38.56%), compared with 194/1,250 machine-generated questions (15.52%).

## Research boundary

ChartQA test is evaluation-only. These 2,500 rows and any later teacher annotations may be used for failure analysis and experimental design, but they must never enter continued QLoRA, DPO, or other optimization. After the teacher taxonomy is validated here, a separate queue must be mined from ChartQA train to create visual-grounding supervision.

## Artifacts

```text
FinChart-R2/results/phase2d_sft_failures_test_2500/
  sft_test_2500_predictions_analyzed.jsonl
  sft_test_2500_failures_676.jsonl
  sft_test_2500_failures_676.json
  sft_test_2500_teacher_priority_after_local_gates.jsonl
  sft_test_2500_failure_report.json
```

Reproduce the analysis with:

```powershell
python FinChart-R2/scripts/analyze_sft_test2500_failures.py
```
