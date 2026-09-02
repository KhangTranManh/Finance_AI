# Protocol-matched SFT versus SFT+DPO comparison

## Outcome

The SFT-only adapter `Kxck/Finance_500_v1` and cumulative provisional DPO adapter `Kxck/Finance_500_v1_DPO_386_provisional` were merged separately into the same Qwen3-VL-4B BF16 base and served through the same vLLM 0.28 OpenAI-compatible path. Both runs use the same prompt, image inputs, greedy decoding, 64-token limit, and exact/numeric matcher.

| Split | SFT | SFT+DPO | DPO delta |
| --- | ---: | ---: | ---: |
| ChartQA `val[0:500]` | 327/500 (65.4%) | 322/500 (64.4%) | -5 / -1.00 pp |
| ChartQA `test[0:2500]` | 1,824/2,500 (72.96%) | 1,822/2,500 (72.88%) | -2 / -0.08 pp |
| Combined, descriptive only | 2,151/3,000 (71.70%) | 2,144/3,000 (71.47%) | -7 / -0.23 pp |

The DPO adapter does not improve overall accuracy. The test delta is practically negligible and not statistically significant, while the validation delta is larger but still based on only five discordant losses.

## Paired transitions

| Split | Both correct | SFT only correct | DPO only correct | Both wrong | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 322 | 5 | 0 | 173 | 0.0625 |
| Test | 1,812 | 12 | 10 | 666 | 0.8318 |
| Combined, descriptive only | 2,134 | 17 | 10 | 839 | 0.2478 |

On test, DPO changes only 22/2,500 correctness outcomes: it fixes ten SFT errors but regresses twelve SFT-correct cases. On validation, it fixes none and regresses five.

## Category-level effect

Across all 3,000 examples, the heuristic task deltas are:

| Task | SFT accuracy | DPO accuracy | Correct-count delta |
| --- | ---: | ---: | ---: |
| Numerical reasoning | 56.99% | 56.29% | -6 |
| Counting | 73.60% | 73.25% | -2 |
| Logical reasoning | 66.23% | 65.58% | -1 |
| Visual grounding / lookup | 80.32% | 80.46% | +2 |

DPO produces a small positive visual-grounding signal but loses more numerical cases than it gains. On test specifically, visual grounding gains three net correct answers, while numerical reasoning loses three and counting loses two.

The most notable operation regression is median: 10/18 correct with SFT versus 6/18 with DPO. Sum also decreases from 42/97 to 40/97. Ratio and average—two major SFT weaknesses—do not improve at all.

## Output-format behavior

| Diagnostic | SFT | SFT+DPO |
| --- | ---: | ---: |
| Structured responses despite final-answer-only prompt | 128 | 145 |
| Completions reaching the 64-token limit | 68 | 74 |

DPO slightly increases output-format drift and truncation, consistent with the preference data retaining structured training-style responses.

## Interpretation

The full 2,500-case test run confirms the smaller validation result: the current 386-pair provisional DPO recipe does not produce a general accuracy improvement. Its net effect is close to zero on test, with a small trade from numerical/counting performance toward visual lookup. The exact McNemar tests do not support a statistically significant difference on either split.

The earlier standalone DPO evaluation produced 324/500 (64.8%). That historical result used a direct in-process vLLM generation path. The primary matched comparison is now the HTTP-serving rerun at 322/500 because it uses the exact same serving client as SFT.

Task and operation categories are deterministic routing heuristics, not teacher-verified causal labels. Both validation and test remain evaluation-only.

## Artifacts

```text
FinChart-R2/results/vali/dpo_vllm_chartqa_3000/
  dpo_vllm_chartqa_val500_test2500_predictions.jsonl
  dpo_vllm_chartqa_val500_test2500_predictions.json
  dpo_vllm_chartqa_val500_test2500_summary.json

FinChart-R2/results/comparison/sft_vs_dpo_val500_test2500/
  sft_vs_dpo_val_paired.jsonl
  sft_vs_dpo_test_paired.jsonl
  sft_vs_dpo_val500_test2500_report.json
```

Reproduce the paired analysis with:

```powershell
python FinChart-R2/scripts/compare_sft_dpo_val500_test2500.py
```
