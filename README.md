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

#### Base-versus-SFT transition analysis

The per-sample comparator assigns one of four deterministic transition tags to every validation example:

| Transition tag | Meaning | Samples |
| --- | --- | ---: |
| `BOTH_CORRECT` | Base correct, SFT correct | 284 |
| `BASE_WRONG_SFT_CORRECT` | Base wrong, SFT correct | 61 |
| `BASE_CORRECT_SFT_WRONG` | Base correct, SFT wrong | 33 |
| `BOTH_WRONG` | Base wrong, SFT wrong | 122 |

For the 114 errors confirmed by the Phase 1 semantic audit, the extracted outcome is:

| Error type | Base errors | SFT fixed | SFT still wrong | Regressions* |
| --- | ---: | ---: | ---: | ---: |
| Numerical reasoning | 54 | 32 | 22 | 16 |
| Counting | 21 | 10 | 11 | 7 |
| Visual extraction | 34 | 8 | 26 | 5 |
| Logical reasoning | 5 | 0 | 5 | 5 |

\*Regression type is a transparent question-text heuristic, not a Phase 1 semantic-audited label, because the base answer was correct for those examples.

**Assessment:** SFT-408 produces a meaningful early gain, driven primarily by numerical-reasoning fixes and with a useful counting improvement. Visual extraction remains the largest unresolved confirmed-error group, while logical reasoning received too little useful supervision in this pilot to improve. The result supports an exact-protocol rerun and a larger, more balanced dataset with extra visual-grounding and logical-reasoning examples; it does not yet justify claiming a final general-purpose improvement.

### Phase 2C — Visual Grounding Diagnosis

Phase 2C is a diagnosis stage before adding a detector, a grounding-aware VLM, or new supervision. It asks whether the visual/counting errors that remain after SFT are truly localization failures and whether their subtypes are concentrated enough to justify grounding-specific training.

```text
Base + SFT frozen validation results
-> BOTH_WRONG and SFT-regression candidates
-> visual/counting triage queue
-> manual + teacher subtype review
-> grounding-supervision decision
```

The current queue contains **78 review candidates**: 37 originate from Phase 1 confirmed visual/counting errors and 41 are question-text heuristic candidates. Each candidate is linked to its ChartQA validation image index and has an explicitly pending final subtype. The review taxonomy includes wrong series/color/category/value/point, legend association, axis alignment, counting, extrema, crop/small text, and other visual failures.

The initial proposed subtype counts are triage only, not training labels. Read [the Phase 2C diagnosis protocol](FinChart-R2/docs/phase2c_visual_grounding_diagnosis.md) before using any reviewed examples for supervision.

### Phase 3 — Reasoning optimization (planned)

After SFT demonstrates the desired solving behavior, the planned next direction is vision GRPO. The objective is to reinforce correct evidence selection, operation choice, numerical consistency, and robust behavior on difficult reasoning cases.

```text
Base 4B -> SFT 4B -> SFT + GRPO 4B
```

Phase 3 is planned research work, not an implemented result.

#### Current Phase 2C direction - train-only preference mining and DPO

The original 51 best-clean visual preference pairs were derived from the frozen validation analysis. They were used only for a leaky DPO infrastructure diagnostic: the adapter may be inspected for training stability, but it is not evaluated on val[0:500] and its output is not compared with the 69.0% SFT result. The earlier ORPO path was retired because its installed TRL/Unsloth stack did not preserve chart-image tensors through preference training.

The reportable Phase 2C path is:

~~~text
ChartQA train[500:2500] -> SFT-408 deterministic inference
-> incorrect-prediction JSONL queue -> teacher validation + manual audit
-> schema-matched DPO pairs -> multimodal DPO -> frozen evaluation
~~~

[Notebook 04](FinChart-R2/notebooks/04_FinChart_R2_Phase2C_DPO_Train_Preference_Mining_Colab.ipynb) mines 2,000 train-only examples by default, resumes safely from JSONL checkpoints, and exports all, errors, and correct JSONL files. The error queue is a candidate pool, not DPO data: teacher review must produce matched chosen and rejected response schemas before training. [Notebook 05](FinChart-R2/notebooks/05_FinChart_R2_Phase2C_DPO_Colab_Leakage_Gated.ipynb) remains a guarded multimodal DPO diagnostic template.

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
- [Phase 2C train-only DPO preference-mining notebook](FinChart-R2/notebooks/04_FinChart_R2_Phase2C_DPO_Train_Preference_Mining_Colab.ipynb)
- [Phase 2C guarded multimodal DPO notebook](FinChart-R2/notebooks/05_FinChart_R2_Phase2C_DPO_Colab_Leakage_Gated.ipynb)
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
| Phase 2C validation-derived DPO diagnostic | Complete infrastructure run; not benchmarkable |
| Phase 2C train-only preference mining | Ready: Notebook 04 mines ChartQA train[500:2500] |
| Phase 2C teacher audit and schema-matched DPO pairs | Next |
| Phase 2C train-only multimodal DPO + frozen evaluation | Pending mined/audited pairs |
| Phase 2 scale-up to 2k–3k+ | Pending pilot result |
| Phase 3 vision GRPO | Planned |

## Experimental guardrails

- The Phase 1 validation subset remains frozen for reportable comparisons. The historical 51-pair validation-derived DPO run is explicitly diagnostic-only and is never evaluated on the same frozen subset.
- Teacher annotations are supervision candidates, not automatic ground truth.
- Questionable labels are not silently rewritten.
- Generated training data is quality-filtered and audited before use.
- Structured supervision is concise and inspectable rather than unrestricted chain-of-thought.
- Scaling decisions depend on frozen evaluation and failure-mode reduction, not only aggregate accuracy.
- Final deployment remains independent of the larger offline teacher model.

## Research hypothesis

FinChart tests whether failure analysis, teacher-assisted structured supervision, targeted QLoRA SFT, and later reasoning optimization can progressively improve chart reasoning reliability in a compact 4B VLM—without requiring a much larger model at inference time.

Read the [Phase 2B pilot evaluation report](reports/phase2b_pilot_408_evaluation.md) for the tracked result summary, limitations, and next decision gate.
