# Phase 2B SFT-408 vLLM evaluation

## Outcome

The SFT-only adapter `Kxck/Finance_500_v1` was merged into the official Qwen3-VL-4B-Instruct BF16 base and evaluated through vLLM with deterministic decoding. The run contains two independently reported ChartQA evaluation slices:

| Split | Range | Correct | Accuracy |
| --- | --- | ---: | ---: |
| Validation | `val[0:500]` | 327/500 | 65.4% |
| Test | `test[0:2500]` | 1,824/2,500 | 72.96% |
| Combined, descriptive only | 3,000 examples | 2,151/3,000 | 71.70% |

The split scores are the primary results. The combined number is descriptive only because validation and test are distinct official splits. No ChartQA training rows were included.

## Protocol-matched DPO comparison

The SFT and provisional DPO adapters now have a direct comparison on the same frozen validation rows, merged-BF16 vLLM backend, prompt, decoding settings, and deterministic matcher:

| Model | Correct | Accuracy | Delta from SFT |
| --- | ---: | ---: | ---: |
| SFT-408 | 327/500 | 65.4% | - |
| DPO-386 provisional, matched HTTP rerun | 322/500 | 64.4% | -5 answers / -1.0 pp |

At the per-example level, both models answer the same 322 validation cases correctly, SFT alone answers five cases correctly, DPO alone answers zero, and both fail 173. On the larger test split, DPO fixes ten SFT failures but regresses twelve SFT-correct cases. Therefore the current 386-pair provisional DPO run does not improve SFT under the matched HTTP-serving comparison.

The earlier Phase 2B notebook result of 345/500 (69.0%) remains historical evidence from a different 4-bit Unsloth generation path. It must not be substituted for the protocol-matched vLLM SFT score when isolating the DPO effect.

## Evaluation contract

- Dataset: `HuggingFaceM4/ChartQA`
- Evaluation only: `val[0:500]` and `test[0:2500]`
- Adapter: `Kxck/Finance_500_v1`
- Adapter revision: `23fee4df75059910aeff8633833474403cc6991a`
- Engine: vLLM 0.28.0 with a merged BF16 checkpoint
- Prompt SHA-256: `94c8753093bda64f29750593ae5522b67b40fe964f72026f38a99a1695bf763f`
- Decoding: greedy, temperature 0, maximum 64 completion tokens
- Matcher: normalized exact match or numeric equality with `1e-6` tolerance
- Completion status: 2,932 `stop`, 68 `length`

The 3,000 prediction records were validated locally: every requested split/index key is present exactly once, the JSON and JSONL exports agree, and an independent matcher recomputation reproduces 327 validation and 1,824 test correct answers.

## Output-format diagnostic

The SFT model does not always follow the final-answer-only request. It emits structured training fields in 24 validation responses and 104 test responses; all 128 fail the unchanged exact/numeric matcher. Of these and other long outputs, 13 validation and 55 test responses reach the 64-token limit, and all 68 are incorrect. This is a real deployment/evaluation issue rather than a reason to alter the frozen score after generation. Future training should separate concise reasoning supervision from inference-format control, or explicitly evaluate a structured-answer parser as a different protocol.

## Local artifacts

Generated predictions are stored under:

```text
FinChart-R2/results/vali/sft_vllm_chartqa_3000/
  sft_vllm_chartqa_val500_test2500_predictions.jsonl
  sft_vllm_chartqa_val500_test2500_predictions.json
  sft_vllm_chartqa_val500_test2500_summary.json
```

The merged checkpoint is intentionally not copied into the repository. It can be reconstructed from the official base and the published SFT adapter with `FinChart-R2/scripts/merge_sft_for_vllm.py`.
