# FinChart-R2

FinChart-R2 contains Phase 2 of the FinChart project: teacher-assisted SFT data construction, targeted multimodal QLoRA training, train-only preference mining, and a completed small-data diagnostic DPO experiment.

## Current status

| Stage | Status | Output |
| --- | --- | --- |
| Phase 2A annotation pilot | Complete | 408 strict-clean ChartQA train examples |
| Phase 2B SFT-408 | Complete | `Kxck/Finance_500_v1` |
| Phase 2B deterministic test | Complete, preliminary | 345/500 (69.0%) on frozen `val[0:500]` |
| SFT merged-BF16 vLLM evaluation | Complete | 327/500 (65.4%) val; 1,824/2,500 (72.96%) test |
| Phase 2C train mining | Complete | 2,000 predictions; 906 error candidates |
| Phase 2C teacher capture | Complete | 906 raw teacher records |
| Phase 2C pair construction | Complete, provisional | 386 candidate DPO pairs |
| Manual DPO-pair audit | Pending | all 386 training pairs remain provisional |
| Multimodal DPO training | Complete, diagnostic | `Kxck/Finance_500_v1_DPO_386_provisional` |
| SFT+DPO matched vLLM evaluation | Complete, diagnostic | 322/500 (64.4%) val; 1,822/2,500 (72.88%) test |

The Phase 2B result is a positive pilot signal, not a final apples-to-apples benchmark claim: its generation prompt and token budget differed from Phase 1. The tracked [evaluation report](../reports/phase2b_pilot_408_evaluation.md) records the limitation and exact numbers.

The Phase 2C adapter was trained for one epoch on 386 automatically gated pairs, split into 347 train and 39 preference-evaluation examples. The exact HTTP-serving comparison now covers both official slices. SFT versus DPO scores are 327/500 (65.4%) versus 322/500 (64.4%) on validation and 1,824/2,500 (72.96%) versus 1,822/2,500 (72.88%) on test. On test, DPO fixes ten cases and regresses twelve; it does not improve overall accuracy. The earlier SFT notebook result of 345/500 (69.0%) and standalone DPO result of 324/500 (64.8%) remain historical evidence from different execution paths. See the [matched comparison](../reports/phase2c_sft_dpo_val500_test2500_comparison.md), [SFT report](../reports/phase2b_sft_vllm_chartqa_3000_evaluation.md), and [standalone DPO report](../reports/phase2c_dpo_386_vllm_evaluation.md).

## Reproducible notebooks

Run the notebooks in this order when rebuilding the current pipeline:

1. [Phase 2A teacher-assisted annotation](notebooks/02_FinChart_R2_Phase2A_Teacher_Assisted_Annotation.ipynb)
2. [Phase 2B SFT-408](notebooks/03_FinChart_R2_Phase2B_Pilot_SFT_408.ipynb)
3. [Phase 2C Kaggle train mining](notebooks/04_FinChart_R2_Phase2C_DPO_Train_Preference_Mining_Kaggle.ipynb)
4. [Phase 2C local raw teacher capture](notebooks/06_FinChart_R2_Phase2C_Teacher_Audit_HuggingFace.ipynb)
5. [Teacher-output analysis and provisional DPO export](notebooks/07_FinChart_R2_Phase2C_Teacher_Raw_Analysis_and_DPO_Export.ipynb)
6. [Unsloth multimodal DPO-386 Colab pilot](notebooks/08_FinChart_R2_Phase2C_Multimodal_DPO_386_Colab.ipynb)
7. [Multimodal DPO-386 Kaggle and Hugging Face publish](notebooks/09_FinChart_R2_Phase2C_Unsloth_Multimodal_DPO_386_Kaggle_HF.ipynb)

Notebook 06 retains its historical filename but now runs locally, loads credentials from `.env`, and saves the provider response without discarding schema-incomplete outputs. Notebook 07 performs parsing, normalization, deterministic conflict checks, routing, and pair export.

## Current Phase 2C data flow

```text
ChartQA train[500:2500]
  -> SFT inference with Kxck/Finance_500_v1
  -> dpo_train_error.jsonl (906 candidates)
  -> dpo_train_teacher_raw_capture.jsonl (906 responses)
  -> phase2c_teacher_v1_dpo_candidates_provisional.jsonl (386 pairs)
  -> provisional one-epoch DPO (347 train / 39 preference-eval)
  -> matched vLLM: 322/500 val; 1,822/2,500 test
  -> manual audit + schema cleanup
  -> controlled DPO rerun only if supervision quality improves
```

Do not use `dpo_train_correct.jsonl` as DPO supervision: a correct model prediction by itself has no rejected response and therefore is not a preference pair. Do not train directly from the 906 deterministic error queue either; it includes representation-equivalent answers, teacher/reference conflicts, ambiguities, and capture failures.

### Candidate diversity and DPO readiness

The current 386 pairs contain 386 unique ChartQA train indices and image references. A deterministic diagnostic profiler assigns the following primary categories:

| Heuristic task category | Samples | Share |
| --- | ---: | ---: |
| Numerical reasoning | 139 | 36.0% |
| Counting | 95 | 24.6% |
| Logical reasoning | 92 | 23.8% |
| Visual grounding / lookup | 60 | 15.5% |

Answer types are 271 numeric (70.2%), 92 yes/no (23.8%), and 23 text/category (6.0%). The 386 pairs were enough to run a conservative one-epoch diagnostic, but DPO does not exceed SFT under the matched 3,000-case HTTP-serving comparison. It is lower by five validation answers and two test answers. Because manual audit is pending and only 54 rejected responses match the complete chosen-response schema, the result cannot establish sample count as the only cause. It establishes that this small, provisional, imbalanced preference dataset is not sufficient for an improvement claim. Visual-grounding and open-text coverage should be expanded before another run.

Reproduce this profile with:

```powershell
python scripts/analyze_phase2c_dpo_candidates.py
```

The precedence is: yes/no target -> logical, explicit count intent -> counting, arithmetic intent -> numerical, and remaining lookup/color/category/value intent -> visual grounding. These are heuristic diagnostics, not teacher-verified or manually audited task labels.

### Notebook 08 training design

Notebook 08 continues from `Kxck/Finance_500_v1` using an Unsloth-loaded 4-bit Qwen3-VL policy and native TRL multimodal DPO with the existing trainable PEFT adapter. Unsloth handles the optimized model path; TRL automatically selects its vision preference collator from the dataset's `image` column. The notebook refuses to train unless `pixel_values` survive collation, preventing accidental text-only preference training.

The implemented conservative configuration is one epoch, effective batch size 8, learning rate `2e-6`, sigmoid DPO, and `beta=0.1`. The completed remote run used 347 training pairs and held out 39 pairs for preference-loss evaluation, finishing 44 optimizer steps. Checkpoints and the resulting adapter are published separately from the source repository. The workflow records dataset hashes, model revisions, library versions, manual-audit status, and whether the run is reportable.

The current provisional data has two explicit limitations: all 386 audit statuses are `PENDING`, and only 54 rejected responses contain the same complete four-field signature as their chosen response. `ALLOW_PROVISIONAL_PAIRS=True` and `ALLOW_SCHEMA_MISMATCH=True` therefore produce a diagnostic adapter only. A final run should use a manually approved, schema-reviewed JSONL and set both flags to `False`.

### Notebook 09 Kaggle publishing path

Notebook 09 uses native TRL 0.29 multimodal DPO and loads the official Qwen3-VL base through Transformers/PEFT, quantizing it on-the-fly with bitsandbytes NF4 before attaching `Kxck/Finance_500_v1`. This avoids lost FP4 state in pre-quantized vision DeepStack layers. A guarded collator wrapper extends TRL 0.29's prompt-only `mm_token_type_ids` across chosen/rejected completion tokens, as required by Qwen3-VL M-RoPE. The Unsloth Python package is intentionally excluded because it currently constrains TRL to 0.24, which has no native vision-preference collator. Kaggle's optional TorchAO package is also removed because this run uses bitsandbytes. Only one GPU is exposed to the process to prevent data-parallel replication of 4-bit parameters. The notebook reads `/kaggle/input/datasets/khangkxcp/dpo300/phase2c_teacher_v1_dpo_candidates_provisional.jsonl` and publishes to `Kxck/Finance_500_v1_DPO_386_provisional`. Store a Hugging Face write token in a Kaggle Secret named `HF_TOKEN`; the notebook never embeds or prints it. After training succeeds, `create_repo(..., exist_ok=True)` creates the repository when absent, while `upload_folder` adds a new commit when it already exists.

### Portable vLLM frozen evaluation

On a Linux host with a CUDA GPU and at least 24 GB VRAM, run:

```bash
cd FinChart-R2
bash scripts/run_phase2c_dpo_vllm_frozen_val.sh
```

The launcher creates an isolated environment under `.artifacts/`, downloads the immutable DPO adapter revision, merges both language and vision-tower LoRA tensors into the official Qwen3-VL BF16 base, and evaluates ChartQA `val[0:500]`. It does not read `.env`. Set `HF_TOKEN` in the shell only if Hub authentication is required. The generated checkpoint is approximately 8.3 GiB and remains Git-ignored.

### SFT-only vLLM serving and 3,000-example evaluation

The portable SFT path merges `Kxck/Finance_500_v1` into the official BF16 base, starts a private OpenAI-compatible vLLM endpoint, and evaluates the two official splits separately:

```bash
python scripts/merge_sft_for_vllm.py \
  --output-dir /path/to/merged_finchartsft_bf16

bash scripts/serve_sft_vllm.sh

python scripts/evaluate_sft_vllm_server_chartqa_3000.py \
  --api-key-file /path/to/api_key \
  --output-dir /path/to/results
```

The completed run scores 327/500 (65.4%) on `val[0:500]` and 1,824/2,500 (72.96%) on `test[0:2500]`. Its combined 2,151/3,000 (71.70%) score is descriptive only. The server binds to `127.0.0.1` by default; use an SSH tunnel for remote access instead of exposing the inference endpoint publicly.

## Local setup

```powershell
cd FinChart-R2
Copy-Item .env.template .env
pip install -r requirements.txt
```

Fill `.env` locally. Never commit it. Phase 2A can also be rebuilt with:

```powershell
python scripts/run_phase2a_pilot.py
python scripts/rebuild_phase2a_v2.py
python scripts/approve_and_build_train_v3.py
```

For Phase 2C, the local capture script is resumable:

```powershell
python scripts/capture_phase2c_teacher_raw.py `
  --input results/finchart_r2_phase2c_train_mining/dpo_train_error.jsonl `
  --output-dir results/finchart_r2_phase2c_train_mining
```

Extract the strict SFT visual-grounding diagnosis set, including ChartQA validation images, with:

```powershell
python scripts/extract_sft_visual_grounding_failures.py
```

The output contains 31 evaluation-only cases: 26 confirmed remaining visual-extraction failures and 5 heuristic visual regressions. These frozen validation examples are for diagnosis and visualization only and must never enter SFT or DPO training.

## Repository contents

```text
configs/       Phase 2A and Phase 2B settings
docs/          data strategy and SFT data contract
notebooks/     executable research workflows
scripts/       local pipeline stages and evaluation comparator
results/       local generated artifacts; ignored by Git
```

Only source code, notebooks, configuration, documentation, and aggregate reports belong in Git. Credentials, provider logs, raw teacher messages, generated JSONL datasets, adapters, and checkpoints remain local.

## Data and evaluation rules

- ChartQA `train` is the only training and preference-mining source.
- ChartQA `val[0:500]` remains frozen and evaluation-only.
- The teacher is an offline annotator, not ground truth.
- SFT targets are concise structured answers, not unrestricted chain-of-thought.
- DPO `chosen` and `rejected` responses must use the same output schema.
- The 386 current pairs are provisional until manual audit is complete.
- A reportable post-DPO comparison must reuse the exact frozen evaluator and generation protocol.
