# SFT val-500 versus test-2500 distribution analysis

## Outcome

The same deterministic taxonomy was applied to the merged-BF16 SFT vLLM predictions on ChartQA `val[0:500]` and `test[0:2500]`. The analysis covers task type, answer type, and operation type for every question and for the corresponding failure subsets.

The main imbalance is consistent across both splits: visual-grounding/lookup questions dominate dataset volume, while numerical-reasoning questions dominate errors.

## Task distribution

### Validation: 500 questions, 173 failures

| Task | Total | Share | Failures | Error rate | Failure share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Visual grounding | 227 | 45.4% | 51 | 22.47% | 29.48% |
| Numerical reasoning | 153 | 30.6% | 78 | **50.98%** | **45.09%** |
| Counting | 70 | 14.0% | 25 | 35.71% | 14.45% |
| Logical reasoning | 50 | 10.0% | 19 | 38.00% | 10.98% |

### Test: 2,500 questions, 676 failures

| Task | Total | Share | Failures | Error rate | Failure share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Visual grounding | 1,196 | 47.84% | 229 | 19.15% | 33.88% |
| Numerical reasoning | 698 | 27.92% | 288 | **41.26%** | **42.60%** |
| Counting | 502 | 20.08% | 126 | 25.10% | 18.64% |
| Logical reasoning | 104 | 4.16% | 33 | 31.73% | 4.88% |

Numerical reasoning is overrepresented among failures by 14.49 percentage points on validation and 14.68 points on test. Visual grounding is underrepresented among failures relative to its volume, but still contributes the second-largest absolute failure count.

## Answer-type imbalance

| Split | Numeric | Text | Yes/no |
| --- | ---: | ---: | ---: |
| Validation | 311 (62.2%) | 139 (27.8%) | 50 (10.0%) |
| Test | 1,915 (76.6%) | 481 (19.24%) | 104 (4.16%) |

The test split contains 14.4 percentage points more numeric targets than validation. Numeric answers account for 136/173 validation failures (78.61%) and 552/676 test failures (81.66%). Yes/no coverage is particularly small in test, so conclusions about logical reasoning have higher sampling uncertainty.

## Operation distribution and difficulty

Lookup is the largest operation by volume: 114/500 (22.8%) on validation and 870/2,500 (34.8%) on test. Count is next in test with 502 examples (20.08%). The test split therefore has 12.0 percentage points more lookup and 6.08 points more counting than validation.

The most consistently difficult operations with meaningful support are:

| Operation | Val failures / total | Val error rate | Test failures / total | Test error rate |
| --- | ---: | ---: | ---: | ---: |
| Ratio | 18/24 | 75.00% | 38/62 | 61.29% |
| Average | 20/30 | 66.67% | 72/138 | 52.17% |
| Difference | 19/36 | 52.78% | 82/150 | 54.67% |
| Sum | 10/20 | 50.00% | 45/77 | 58.44% |
| Median | 3/7 | 42.86% | 5/11 | 45.45% |

Product, median, and percentage-change questions are sparse. Their percentages should not drive curriculum decisions without more samples.

## Engineering interpretation

- The 500 and 2,500 slices are not distribution-identical. Report them separately.
- Visual grounding is the dominant input task by volume, but numerical reasoning is the dominant error concentration.
- Numeric supervision is abundant, while logical/yes-no and rare operations are underrepresented.
- A visual teacher should not inspect only lookup questions: wrong visual extraction can propagate into ratio, average, difference, sum, and counting failures.
- For later train-only data construction, use soft sampling weights rather than discarding clean majority groups.

All categories are deterministic question-intent heuristics, not teacher-verified labels or causal error diagnoses. Validation and test records remain evaluation-only and are never eligible for training.

## Artifacts

```text
FinChart-R2/results/phase2d_sft_val500_test2500_distribution/
  sft_val_distribution_labeled.jsonl
  sft_test_distribution_labeled.jsonl
  sft_val500_test2500_distribution_report.json
```

Reproduce with:

```powershell
python FinChart-R2/scripts/compare_sft_val500_test2500_distributions.py
```
