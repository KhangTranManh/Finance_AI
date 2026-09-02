# Phase 2C DPO-386 vLLM evaluation

## Outcome

The provisional adapter `Kxck/Finance_500_v1_DPO_386_provisional` was trained for one epoch and first evaluated on the frozen ChartQA `val[0:500]` subset through a standalone vLLM path. That historical run scored 324/500 correct (64.8%). A later matched HTTP-serving rerun scores 322/500 (64.4%) for DPO versus 327/500 (65.4%) for SFT, confirming that the current small-data DPO experiment does not improve the model.

| Model/run | Correct | Accuracy |
| --- | ---: | ---: |
| Base 4B Phase 1 | 317/500 | 63.4% |
| SFT-408 pilot, historical Unsloth path | 345/500 | 69.0% |
| SFT-408, merged BF16 + vLLM | 327/500 | 65.4% |
| DPO-386 provisional, standalone merged BF16 + vLLM | 324/500 | 64.8% |
| DPO-386 provisional, matched vLLM HTTP server | 322/500 | 64.4% |

On the primary matched HTTP-serving path, the provisional DPO run is lower by five validation examples, or 1.0 percentage point. Both models answer the same 322 cases correctly; SFT alone answers five cases correctly, DPO alone answers zero, and both fail 173. On `test[0:2500]`, DPO scores 1,822 versus SFT's 1,824, fixing ten cases but regressing twelve. In this project, 386 provisional pairs—347 used for optimization—were not sufficient to demonstrate a DPO improvement. This should not be interpreted as proof that dataset size alone caused the regression because the supervision was unaudited and chosen/rejected formats were asymmetric. The earlier 345/500 SFT score used the 4-bit Unsloth path and is retained as historical, non-matched evidence.

## Frozen evaluation contract

- Dataset: `HuggingFaceM4/ChartQA`, `val[0:500]`
- Prompt: exact Phase 1 final-answer-only prompt
- Decoding: greedy, temperature 0, maximum 64 new tokens
- Context limit: 2,048 tokens
- Matcher: normalized exact match or numeric equality with `1e-6` tolerance
- Adapter revision: `2a6df004d4522cb5b7a072fd6b8dea4d54d7b64d`
- Engine: vLLM 0.28.0 on a merged BF16 checkpoint
- Runtime: 28.339 seconds for 500 examples after engine initialization

The DPO adapter contains both language-backbone and vision-tower LoRA tensors. It was merged before inference so vLLM could not silently omit experimental multimodal-tower LoRA weights.

## Diagnostic interpretation

The training run used 386 automatically gated preference pairs: 347 for training and 39 for preference-loss evaluation. All 386 still have `manual_audit_status=PENDING`, so the adapter and score are explicitly provisional.

Output-format drift is visible in the frozen predictions: 31/500 responses contain structured supervision fields even though the evaluator requested only the final answer. All 31 fail the unchanged matcher, and only 16 reach an explicit `Answer:` line within the 64-token limit. This suggests that preference training reinforced the training response schema at the expense of instruction-following under the final-answer-only evaluation contract.

The current evidence does not justify scaling the same DPO recipe or claiming that DPO improved FinChart. SFT and DPO have now been rerun through the same merged-BF16 vLLM HTTP-serving evaluator, isolating the adapter change: DPO is lower by 1.0 point on validation and 0.08 points on test. The next controlled experiment should first audit the preference pairs, make chosen/rejected schemas symmetrical, and increase balanced coverage beyond the current 386 pairs. See the [full matched comparison](phase2c_sft_dpo_val500_test2500_comparison.md).

## Local artifacts

Generated JSON data and training metadata are stored under:

```text
FinChart-R2/results/checkpoints/dpo_386_provisional/
FinChart-R2/results/vali/phase2c_dpo_vllm_val_0_500/
```

The full merged checkpoint is intentionally excluded from the project because it is approximately 8.3 GiB and can be reconstructed from the Hugging Face adapter with `FinChart-R2/scripts/run_phase2c_dpo_vllm_frozen_val.sh`.
