# FinChart

FinChart is a research-oriented engineering project on reliable chart understanding and quantitative visual reasoning with a compact vision-language model. It investigates whether **Qwen3-VL-4B-Instruct** can be specialized for chart-analysis workloads through measured failure analysis, teacher-assisted dataset engineering, targeted multimodal SFT, and later reasoning optimization.

The goal is not simply to maximize one benchmark score. FinChart studies whether a compact model can achieve useful reliability for specialized chart reasoning while reducing dependence on larger, more expensive VLMs at inference time.

## Research question

> How much chart understanding and quantitative reasoning capability can be transferred into a compact vision-language model through targeted supervision and reasoning optimization?

```text
Evaluate -> diagnose failures -> construct targeted supervision -> SFT
         -> frozen re-evaluation -> reasoning optimization -> comparison
```

Rather than training blindly on more data, each training decision is based on observed model failures.

## Why chart reasoning?

Answering a chart question requires a model to connect multiple capabilities:

```text
Question understanding -> visual grounding -> value extraction
                       -> operation selection -> numerical/logical reasoning -> answer
```

An incorrect answer can therefore have different causes: selecting the wrong series, reading the wrong bar, choosing the wrong operation, making an arithmetic error, returning an incompatible representation, or miscounting visual elements. FinChart diagnoses these failure types separately.

## Scope and model

- **Student / deployable model:** `Qwen3-VL-4B-Instruct`
- **Dataset:** `HuggingFaceM4/ChartQA`
- **Training boundary:** ChartQA `train` only
- **Frozen evaluation:** ChartQA `val[0:500]`
- **Teacher role:** offline structured annotation only; the teacher is not used at deployment or treated as ground truth.

The project targets four capabilities: visual grounding, numerical reasoning, counting, and logical reasoning.

## Experimental design

### Phase 1 — Baseline evaluation and failure analysis

Phase 1 evaluates the base model on the frozen 500-example validation subset with a hybrid protocol: deterministic normalized matching followed by a VLM semantic examiner for unresolved mismatches. Verdicts distinguish `CORRECT`, `INCORRECT`, `AMBIGUOUS`, and `POSSIBLE_LABEL_ERROR` rather than treating all string mismatches as model errors.

| Phase 1 result | Value |
| --- | ---: |
| Evaluated samples | 500 |
| Deterministic correct | 317 |
| Deterministic accuracy | 63.4% |
| Final correct | 351 |
| Final incorrect | 114 |
| Resolved coverage | 93.0% |
| Resolved accuracy | 75.48% |

Confirmed error distribution:

| Failure mode | Errors | Share |
| --- | ---: | ---: |
| Numerical reasoning | 54 | 47.4% |
| Visual extraction | 34 | 29.8% |
| Counting | 21 | 18.4% |
| Logical reasoning | 5 | 4.4% |

The main finding is that the base model generally understands chart questions, but remains limited by inconsistent visual grounding, numerical reasoning, and counting. This measured profile defines the Phase 2 curriculum.

### Phase 2A — Teacher-assisted dataset engineering

Phase 2A constructs SFT supervision from ChartQA train examples only.

```text
ChartQA train -> normalization -> VLM teacher -> structured annotation
              -> deterministic validation -> quality filtering/audit -> SFT-ready data
```

The teacher generates closed-taxonomy annotations with fields such as task type, target series/category, relevant values, operation, calculation, final answer, and confidence. Its output is validated through schema and taxonomy checks, answer and representation equivalence, arithmetic recomputation where possible, conflict detection, confidence thresholds, and manual audit.

Samples receive statuses including `VALIDATED`, `REVIEW_REPRESENTATION`, `REVIEW_CONFLICT`, `LOW_CONFIDENCE`, `INVALID_SCHEMA`, and `TEACHER_ERROR`. Only validated, strict-clean examples are eligible for pilot training.

The initial 500-example ChartQA train pilot produced **408 strict-clean examples (81.6%)**, with this distribution:

| Task type | Examples |
| --- | ---: |
| Numerical reasoning | 150 |
| Visual grounding | 144 |
| Logical reasoning | 77 |
| Counting | 37 |

The approved pilot dataset stays local under `FinChart-R2/results/` and is intentionally excluded from Git.

### Phase 2B — Targeted multimodal SFT

Phase 2B fine-tunes Qwen3-VL-4B-Instruct with **Unsloth + QLoRA** using the approved Phase 2A examples. The pilot has completed training and deterministic testing through [the Phase 2B notebook](FinChart-R2/notebooks/03_FinChart_R2_Phase2B_Pilot_SFT_408.ipynb).

- **Input:** chart image + question
- **Target:** concise, inspectable supervision: relevant values, operation, calculation, and final answer
- **Not used:** unrestricted chain-of-thought

The pilot SFT is designed to teach the sequence:

```text
chart/question understanding -> correct grounding -> value extraction
-> operation selection -> reasoning -> answer
```

The completed pilot trained on 408 strict-clean examples for 2 epochs (102 optimizer steps) and saved its QLoRA adapter. Its test on `ChartQA val[0:500]` achieved **345/500 = 69.0% deterministic accuracy**, compared with the Phase 1 base result of **317/500 = 63.4%**: a **+5.6 percentage-point** gain. On the same validation indices, SFT corrected 61 base-model errors and regressed on 33 previously correct cases.

This is a positive pilot result, not a final Phase 2 claim: the completed SFT test used a different generation prompt and token budget from the original frozen R1 run. The next evaluation task is a protocol-identical rerun with the R1 prompt/configuration and semantic error analysis, followed by a decision on 2k–3k+ scale-up.

### Phase 3 — Reasoning optimization (planned)

After SFT demonstrates the desired solving behavior, the planned next direction is vision GRPO. The objective is to reinforce correct evidence selection, operation choice, numerical consistency, and robust behavior on difficult reasoning cases.

```text
Base 4B -> SFT 4B -> SFT + GRPO 4B
```

Phase 3 is planned research work, not an implemented result.

## Decision and scaling strategy

```text
~400 validated examples -> pilot SFT -> frozen evaluation
                                      |
                         improvement? |
                           yes         no
                            |           |
                     scale to 2k-3k+  revise data, target format,
                                      curriculum, or LoRA configuration
```

The project does not scale teacher annotation before the pilot establishes that the supervision format improves the frozen evaluation.

## Repository structure

```text
FinChart/
├── FinChart-R1/   # frozen baseline evaluation and failure analysis
├── FinChart-R2/   # Phase 2A data engineering and Phase 2B SFT
│   ├── configs/
│   ├── docs/
│   ├── notebooks/
│   ├── scripts/
│   ├── results/   # local generated data, ignored by Git
│   └── outputs/   # local adapters/checkpoints, ignored by Git
└── README.md
```

Useful entry points:

- [Phase 1 notebooks](FinChart-R1/notebooks/)
- [Phase 2A teacher-annotation notebook](FinChart-R2/notebooks/02_FinChart_R2_Phase2A_Teacher_Assisted_Annotation.ipynb)
- [Phase 2B pilot SFT notebook](FinChart-R2/notebooks/03_FinChart_R2_Phase2B_Pilot_SFT_408.ipynb)
- [Phase 2 data contract](FinChart-R2/docs/phase2a_data_contract.md)

## Current status

| Stage | Status |
| --- | --- |
| Phase 1 baseline evaluation | Complete |
| Phase 1 failure analysis | Complete |
| Phase 2A teacher-annotation pilot | Complete |
| Phase 2A strict-clean pilot dataset | Complete (408 local examples) |
| Phase 2B pilot QLoRA SFT | Complete: 408 examples, 2 epochs |
| Phase 2B test: ChartQA `val[0:500]` | Complete: 69.0% (345/500), +5.6 pp vs base |
| Protocol-identical frozen rerun + semantic error analysis | Next |
| Phase 2 scale-up to 2k–3k+ | Pending pilot result |
| Phase 3 vision GRPO | Planned |

## Experimental guardrails

- The Phase 1 validation subset remains frozen and is never used for annotation or SFT.
- Teacher annotations are supervision candidates, not automatic ground truth.
- Questionable labels are not silently rewritten.
- Generated training data is quality-filtered and audited before use.
- Structured supervision is concise and inspectable rather than unrestricted chain-of-thought.
- Scaling decisions depend on frozen evaluation and failure-mode reduction, not only aggregate accuracy.
- Final deployment remains independent of the larger offline teacher model.

## Research hypothesis

FinChart tests whether failure analysis, teacher-assisted structured supervision, targeted QLoRA SFT, and later reasoning optimization can progressively improve chart reasoning reliability in a compact 4B VLM—without requiring a much larger model at inference time.

Read the [Phase 2B pilot evaluation report](reports/phase2b_pilot_408_evaluation.md) for the tracked result summary, limitations, and next decision gate.
