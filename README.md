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
| Phase 2B | Unsloth vision QLoRA SFT | 345/500 (69.0%), a preliminary +5.6 percentage-point gain |
| Phase 2C | Train-only preference mining and teacher review | 377 provisional DPO candidates awaiting manual audit |

The Phase 2B comparison used the same validation indices, producing 284 both-correct cases, 61 SFT fixes, 33 regressions, and 122 both-wrong cases. The paired deterministic result is encouraging (`p = 0.0051` by exact McNemar test), but the SFT run used a different generation prompt and token budget from Phase 1. A protocol-identical rerun remains necessary before making a final model-quality claim. See the [Phase 2B evaluation report](reports/phase2b_pilot_408_evaluation.md).

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
-> 377 provisional prompt/chosen/rejected pairs
-> manual audit
-> multimodal DPO
-> frozen val[0:500] evaluation
```

Mining produced 2,000 predictions: 1,094 deterministic matches and 906 error candidates. Teacher analysis routed the 906 candidates as follows:

| Route | Samples |
| --- | ---: |
| Provisional DPO candidate | 377 |
| Representation-equivalent | 178 |
| Model already correct | 122 |
| Teacher/reference conflict | 123 |
| Dataset ambiguity | 64 |
| Capture failure or unparseable | 41 |
| Unknown verdict | 1 |

The 377 candidates passed automated pair gates, but the teacher remains an annotator rather than an oracle. Manual audit is required before training. Their heuristic task mix is 37.9% numerical, 30.0% counting, 21.5% logical, 6.9% lookup/other, and 3.7% explicit visual; visual coverage is therefore the main data-balance weakness.

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
