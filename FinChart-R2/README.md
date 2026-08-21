# FinChart-R2

Phase 2 of FinChart tests whether targeted, teacher-assisted structured supervision can improve chart reasoning in **Qwen3-VL-4B-Instruct**, while keeping the deployable model compact. The teacher is used offline only to create supervision; it is never required at inference time.

## Research scope

Phase 1 froze the evaluator and ChartQA `val[0:500]` subset, then identified four principal failure modes: numerical reasoning, visual grounding/value extraction, counting, and logical reasoning. Phase 2 uses **only ChartQA train** for SFT; the Phase 1 validation subset remains training-free.

Phase 2A runs a teacher-assisted data-engineering pipeline:

```text
ChartQA train -> normalization -> VLM teacher -> structured annotation
-> deterministic validation -> quality filtering/audit -> SFT-ready data
```

Each annotation has a closed task taxonomy and fields for target series/category, relevant values, operation, calculation, answer, and confidence. Only `VALIDATED` examples that pass strict filtering are eligible for the pilot SFT dataset. The current local pilot output contains 408 strict-clean examples from a 500-example annotation run; it is intentionally not committed to Git.

Phase 2B has trained a QLoRA pilot on the 408 approved examples. Its preliminary deterministic validation result is 69.0% (345/500), versus the 63.4% Phase 1 base score. The result is tracked as preliminary because the pilot generation prompt differs from the frozen Phase 1 prompt; a protocol-identical rerun and semantic error analysis remain required before scaling data.

## Phase 2C: train-only preference mining for multimodal DPO

Phase 2C focuses on residual visual-grounding and counting failures. The previous 51 validation-derived preference pairs are retained only as a leaky infrastructure diagnostic; they cannot produce a reportable frozen-validation result. ORPO is retired in this project because the installed ORPO path did not preserve image tensors through preference training.

The next reportable path mines new candidates only from ChartQA train:

~~~text
ChartQA train[500:2500] -> SFT-408 inference -> deterministic error queue
-> teacher validation + manual audit -> schema-matched DPO pairs
-> multimodal DPO -> frozen ChartQA val[0:500] evaluation
~~~

Run [Notebook 04](notebooks/04_FinChart_R2_Phase2C_DPO_Train_Preference_Mining_Colab.ipynb) on Colab to create resumable JSONL outputs under Google Drive. It defaults to 2,000 examples and writes all predictions, incorrect candidates, correct predictions, and a manifest. Incorrect predictions are review candidates only: do not train them directly. After teacher/audit review creates matched prompt / chosen / rejected pairs from ChartQA train, use [Notebook 05](notebooks/05_FinChart_R2_Phase2C_DPO_Colab_Leakage_Gated.ipynb) as the multimodal DPO template.

## Repository layout

```text
FinChart-R2/
|- configs/       # non-secret reproducible settings
|- docs/          # research design and data contract
|- notebooks/     # Colab entry point (outputs stripped for Git)
|- scripts/       # reproducible Phase 2A pipeline stages
|- results/       # local generated data; ignored by Git
|- .env.template  # copy to .env and fill locally
`- requirements.txt
```

## Quick start: Phase 2A

1. Create a local environment file: `Copy-Item .env.template .env`, then set the teacher credentials locally.
2. Install dependencies: `pip install -r requirements.txt`.
3. In Colab, upload and run [the Phase 2A notebook](notebooks/02_FinChart_R2_Phase2A_Teacher_Assisted_Annotation.ipynb), or run the local stages in order:

   ```powershell
   python scripts/run_phase2a_pilot.py
   python scripts/rebuild_phase2a_v2.py
   python scripts/approve_and_build_train_v3.py
   ```

Generated annotations, logs, audits, and SFT data stay under `results/` and are excluded from version control.

## Phase 2B pilot SFT

Use [the Phase 2B pilot notebook](notebooks/03_FinChart_R2_Phase2B_Pilot_SFT_408.ipynb) after Phase 2A produces the approved `v3_train_clean` JSONL. It trains the QLoRA pilot and saves adapters, checkpoints, metrics, and predictions locally. The tracked [pilot report](../reports/phase2b_pilot_408_evaluation.md) contains the extracted result; raw artifacts are intentionally ignored.

## Phase 2C quick start

1. In Colab, select Runtime Version 2026.07 and a GPU runtime.
2. Run [Notebook 04](notebooks/04_FinChart_R2_Phase2C_DPO_Train_Preference_Mining_Colab.ipynb) through its JSONL export cell.
3. Review and annotate only the error JSONL; retain subtype, evidence, and quality metadata.
4. Build equal-schema DPO preference pairs from the validated train-only candidates.
5. Run multimodal DPO, then evaluate once with the unchanged frozen Phase 1 evaluator.

## Guardrails

- Never train on the frozen Phase 1 validation subset.
- The teacher is a primary annotator, not ground truth.
- Validate schema, taxonomy, answer representation, arithmetic, confidence, and semantic conflicts before SFT.
- Do not use unrestricted chain-of-thought; targets are concise, inspectable structured supervision.
- Scale from 500 to 2k-3k+ examples only after frozen evaluation shows a pilot improvement.
- Do not report a score from the validation-derived 51-pair DPO diagnostic.
