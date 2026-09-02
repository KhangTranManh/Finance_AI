# FinChart

FinChart is a research and engineering project for improving chart question answering in a compact vision-language model. The project follows an evidence-driven loop:

```text
frozen baseline -> failure analysis -> targeted supervision
-> compact-model training -> frozen re-evaluation
```

The deployable model is **Qwen3-VL-4B-Instruct**. Larger teacher models are used offline to propose and review training supervision; they are not required at inference time and are not treated as ground truth.

## Research hypothesis

FinChart tests whether targeted, teacher-assisted structured supervision derived from measured baseline failures can improve the chart-reasoning capability of a compact 4B vision-language model without relying on a much larger model at inference time.

## Experimental boundary

- Dataset: `HuggingFaceM4/ChartQA`
- Training and preference mining: ChartQA `train` only
- Frozen benchmark: ChartQA `val[0:500]`
- Main failure families: numerical reasoning, visual grounding, counting, and logical reasoning
- Supervision format: concise relevant values, operation, calculation, and answer; no unrestricted chain-of-thought

## Progress and results

| Phase | Method | Current result |
| --- | --- | --- |
| Phase 1 | Base-model evaluation and failure analysis | 317/500 deterministic correct (63.4%) |
| Phase 2A | Teacher-assisted dataset engineering | 408 strict-clean SFT examples from a 500-example train pilot |
| Phase 2B | Unsloth vision QLoRA SFT | Historical 345/500 (69.0%); matched vLLM 327/500 (65.4%) |
| Phase 2C | Small-data provisional multimodal DPO | 322/500 (64.4%) val; 1,822/2,500 (72.88%) test |

The original Phase 2B comparison used the same validation indices, producing 284 both-correct cases, 61 SFT fixes, 33 regressions, and 122 both-wrong cases. The paired deterministic result was encouraging (`p = 0.0051` by exact McNemar test), but that SFT run used a different generation prompt and token budget from Phase 1. A later merged-BF16 vLLM rerun provides the protocol-matched SFT control for DPO and scores 327/500 (65.4%). See the [original Phase 2B report](reports/phase2b_pilot_408_evaluation.md) and [SFT vLLM report](reports/phase2b_sft_vllm_chartqa_3000_evaluation.md).

## Phase 2 workflow

### Phase 2A: teacher-assisted dataset engineering

```text
ChartQA train -> normalization -> vision-language teacher
-> structured annotation -> deterministic checks -> quality audit
-> strict-clean SFT dataset
```

The teacher uses closed task and operation taxonomies. Schema validation, answer equivalence, arithmetic checks, confidence filtering, conflict detection, and manual audit decide whether an annotation is eligible for training.

### Phase 2B: targeted multimodal SFT

Qwen3-VL-4B-Instruct was fine-tuned with Unsloth and QLoRA on 408 approved examples. The released adapter used by later mining is `Kxck/Finance_500_v1`.

### Phase 2C: train-only preference data

The current reportable Phase 2C pipeline avoids validation leakage:

```text
ChartQA train[500:2500]
-> SFT-408 inference
-> 906 deterministic error candidates
-> offline teacher raw capture
-> deterministic normalization and conflict gates
-> 386 provisional prompt/chosen/rejected pairs
-> one-epoch diagnostic multimodal DPO (347 train / 39 preference-eval)
-> matched vLLM evaluation: 322/500 val; 1,822/2,500 test
-> manual audit and controlled rerun still required
```

Mining produced 2,000 predictions: 1,094 deterministic matches and 906 error candidates. Teacher analysis routed the 906 candidates as follows:

| Route | Samples |
| --- | ---: |
| Provisional DPO candidate | 386 |
| Representation-equivalent | 197 |
| Model already correct | 129 |
| Teacher/reference conflict | 128 |
| Dataset ambiguity | 65 |
| Unknown verdict | 1 |

The 386 candidates passed automated pair gates, but the teacher remains an annotator rather than an oracle. A provisional diagnostic run was completed before manual audit; manual review is still required before these pairs can support a reportable or final training run.

### Current DPO candidate categories

All 386 candidates have unique ChartQA train indices and unique image references. The category profile is deterministic and reproducible, but heuristic: the teacher export does not contain a manually validated `task_type`, so these labels are dataset diagnostics rather than ground truth.

| Primary task category | Samples | Share |
| --- | ---: | ---: |
| Numerical reasoning | 139 | 36.0% |
| Counting | 95 | 24.6% |
| Logical reasoning | 92 | 23.8% |
| Visual grounding / lookup | 60 | 15.5% |

| Normalized operation | Samples | Share |
| --- | ---: | ---: |
| Multi-step | 95 | 24.6% |
| Count | 90 | 23.3% |
| Average | 36 | 9.3% |
| Lookup | 31 | 8.0% |
| Comparison | 26 | 6.7% |
| Difference | 25 | 6.5% |
| Sum | 23 | 6.0% |
| Extrema | 19 | 4.9% |
| Median | 16 | 4.1% |
| Ratio | 14 | 3.6% |
| Other | 7 | 1.8% |
| Product | 4 | 1.0% |

Answer targets are 271 numeric (70.2%), 92 yes/no (23.8%), and 23 text/category answers (6.0%). This small dataset was sufficient to execute a diagnostic DPO pilot, but it did not improve the matched SFT result. On the identical HTTP-serving path, validation changes from 327/500 (65.4%) to 322/500 (64.4%), while test changes from 1,824/2,500 (72.96%) to 1,822/2,500 (72.88%). The result does not prove that sample count alone caused the decline because the pairs are unaudited and chosen/rejected schemas are asymmetric. It does show that the current 386-pair recipe is not ready to scale or claim as an improvement. Visual-grounding and open-text answers also remain underrepresented.

[Notebook 08](FinChart-R2/notebooks/08_FinChart_R2_Phase2C_Multimodal_DPO_386_Colab.ipynb) defines the Colab alternative: Unsloth 4-bit Qwen3-VL loading, continuation from the SFT-408 adapter, native TRL vision-DPO collation, an image-tensor preflight, resumable one-epoch DPO, Drive checkpoints, and an optional Hugging Face upload. Its default 386-pair mode is explicitly diagnostic because manual audit is still pending and only 54 rejected responses currently match the full four-field chosen-response schema.

[Notebook 09](FinChart-R2/notebooks/09_FinChart_R2_Phase2C_Unsloth_Multimodal_DPO_386_Kaggle_HF.ipynb) provides the Kaggle training and publishing path. It uses native TRL 0.29 multimodal DPO, quantizes the official Qwen3-VL base on-the-fly with bitsandbytes NF4, attaches the SFT-408 adapter, reads a write-scoped `HF_TOKEN` from Kaggle Secrets, loads the fixed `khangkxcp/dpo300` JSONL input, and creates or updates `Kxck/Finance_500_v1_DPO_386_provisional` after successful training.

The provisional DPO adapter has now been trained and published. Under the same merged-BF16 vLLM HTTP-serving protocol, SFT versus DPO scores are 327 versus 322 correct on validation and 1,824 versus 1,822 on test. DPO therefore does **not** improve overall accuracy. On test it fixes ten SFT errors but regresses twelve SFT-correct cases; its small visual-grounding gain is offset by numerical and counting losses. The historical SFT notebook score of 345/500 (69.0%) used a different inference path and remains preliminary rather than the matched control. See the [full SFT/DPO comparison](reports/phase2c_sft_dpo_val500_test2500_comparison.md), [SFT vLLM report](reports/phase2b_sft_vllm_chartqa_3000_evaluation.md), and [historical DPO-500 report](reports/phase2c_dpo_386_vllm_evaluation.md).

## Repository map

```text
FinChart-R1/             Phase 1 frozen baseline and error analysis
FinChart-R2/
  configs/              reproducible Phase 2 settings
  docs/                 data strategy and contracts
  notebooks/            Phase 2A, SFT, mining, capture, and pair export
  scripts/              local reproducible pipeline stages
  results/              generated local data, ignored by Git
reports/                 tracked experiment summaries
```

Start with the [FinChart-R2 guide](FinChart-R2/README.md). Generated datasets, raw teacher messages, checkpoints, and credentials are intentionally excluded from Git.

## Guardrails

- Never use the frozen validation subset for training or preference optimization.
- Do not convert every deterministic mismatch directly into a preference pair.
- Keep raw teacher output for audit, but validate it independently before training.
- Compare models with the same evaluator, prompt, decoding settings, and validation indices.
- Report aggregate gains together with regressions and failure-mode changes.
