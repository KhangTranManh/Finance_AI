# Phase 2B pilot SFT evaluation

## Experiment summary

| Item | Value |
| --- | --- |
| Student model | `unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit` |
| SFT method | Unsloth vision QLoRA |
| LoRA configuration | rank 16, alpha 16 |
| Training data | 408 Phase 2A strict-clean ChartQA train examples |
| Training schedule | 2 epochs, 102 optimizer steps, learning rate `1e-4` |
| Train loss | 0.7323 |
| Training/evaluation separation | ChartQA train for SFT; ChartQA `val[0:500]` for evaluation |

The raw adapter, checkpoints, training metrics, and per-example predictions remain local and are excluded from Git. The input prediction file for this report is `FinChart-R2/results/vali/sft_val_500.csv`.

## Deterministic validation result

The pilot generated answers for all 500 frozen validation instances. Re-scoring its saved predictions with the Phase 1 deterministic matcher produced the same score as the pilot CSV.

| Model | Correct | Total | Deterministic accuracy |
| --- | ---: | ---: | ---: |
| Phase 1 base Qwen3-VL-4B | 317 | 500 | 63.4% |
| Phase 2B SFT-408 pilot | 345 | 500 | 69.0% |
| Difference | +28 | — | **+5.6 pp** |

Paired sample transitions on the same validation indices:

| Transition | Samples |
| --- | ---: |
| Correct for both models | 284 |
| Correct only for base | 33 |
| Correct only for SFT | 61 |
| Incorrect for both models | 122 |

The paired McNemar exact test gives `p = 0.0051`. This is a positive pilot signal: the SFT model fixes more baseline mistakes than it introduces regressions.

## Observed limitations

- The saved SFT evaluation used a bare-question prompt and `max_new_tokens=192`; Phase 1 used its frozen final-answer prompt and `max_new_tokens=64`. Therefore the 69.0% result is **promising preliminary evidence**, not yet the final protocol-identical Phase 1 comparison.
- SFT predictions are deterministically parsed from the concise structured response before matching. Eight predictions leaked structured or multiline fields and all eight were marked incorrect.
- Numeric ground-truth items scored 194/311 (62.4%), while text answers scored 151/189 (79.9%). Numerical reasoning remains the primary improvement opportunity.
- The pilot evaluation cell loaded the adapter identifier `Kxck/Finance_500_v1`. A future protocol-identical rerun should load the local saved pilot adapter or record the exact Hub revision hash.
- The semantic examiner and failure-mode breakdown from Phase 1 have not yet been re-run for the SFT adapter.

## Decision

The 408-example pilot is sufficient to justify evaluator standardization and a protocol-identical frozen rerun. It is not sufficient by itself to claim final Phase 2 success or to scale annotation automatically. After the exact rerun and error-category analysis, scale to 2k–3k+ validated examples only if the improvement holds.
