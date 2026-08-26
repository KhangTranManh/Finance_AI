# FinChart-R2

FinChart-R2 contains Phase 2 of the FinChart project: teacher-assisted SFT data construction, targeted multimodal QLoRA training, and train-only preference mining for a later DPO experiment.

## Current status

| Stage | Status | Output |
| --- | --- | --- |
| Phase 2A annotation pilot | Complete | 408 strict-clean ChartQA train examples |
| Phase 2B SFT-408 | Complete | `Kxck/Finance_500_v1` |
| Phase 2B deterministic test | Complete, preliminary | 345/500 (69.0%) on frozen `val[0:500]` |
| Phase 2C train mining | Complete | 2,000 predictions; 906 error candidates |
| Phase 2C teacher capture | Complete | 906 raw teacher records |
| Phase 2C pair construction | Complete, provisional | 377 candidate DPO pairs |
| Manual DPO-pair audit | Next | required before preference training |
| Multimodal DPO and frozen evaluation | Pending | run only after audit |

The Phase 2B result is a positive pilot signal, not a final apples-to-apples benchmark claim: its generation prompt and token budget differed from Phase 1. The tracked [evaluation report](../reports/phase2b_pilot_408_evaluation.md) records the limitation and exact numbers.

## Reproducible notebooks

Run the notebooks in this order when rebuilding the current pipeline:

1. [Phase 2A teacher-assisted annotation](notebooks/02_FinChart_R2_Phase2A_Teacher_Assisted_Annotation.ipynb)
2. [Phase 2B SFT-408](notebooks/03_FinChart_R2_Phase2B_Pilot_SFT_408.ipynb)
3. [Phase 2C Kaggle train mining](notebooks/04_FinChart_R2_Phase2C_DPO_Train_Preference_Mining_Kaggle.ipynb)
4. [Phase 2C local raw teacher capture](notebooks/06_FinChart_R2_Phase2C_Teacher_Audit_HuggingFace.ipynb)
5. [Teacher-output analysis and provisional DPO export](notebooks/07_FinChart_R2_Phase2C_Teacher_Raw_Analysis_and_DPO_Export.ipynb)

Notebook 06 retains its historical filename but now runs locally, loads credentials from `.env`, and saves the provider response without discarding schema-incomplete outputs. Notebook 07 performs parsing, normalization, deterministic conflict checks, routing, and pair export.

## Current Phase 2C data flow

```text
ChartQA train[500:2500]
  -> SFT inference with Kxck/Finance_500_v1
  -> dpo_train_error.jsonl (906 candidates)
  -> dpo_train_teacher_raw_capture.jsonl (906 responses)
  -> phase2c_teacher_v1_dpo_candidates_provisional.jsonl (377 pairs)
  -> manual audit
  -> final DPO dataset
```

Do not use `dpo_train_correct.jsonl` as DPO supervision: a correct model prediction by itself has no rejected response and therefore is not a preference pair. Do not train directly from the 906 deterministic error queue either; it includes representation-equivalent answers, teacher/reference conflicts, ambiguities, and capture failures.

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
- The 377 current pairs are provisional until manual audit is complete.
- A reportable post-DPO comparison must reuse the exact frozen evaluator and generation protocol.
